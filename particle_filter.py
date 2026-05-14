import random
import difflib
import osmnx as ox
import time
import re
from shapely.geometry import Point, LineString
import numpy as np
from pyproj import Transformer


class Particle:
    def __init__(self, start: int, end: int, dist: float, direction: int, max_length: float, weight: float):
        """ Represents a particle on a graph edge.
        Args:
            start (int): start node id of edge
            end (int): end node id of edge
            dist (float): distance along edge in meters
            direction (int): movement direction (1 or -1)
            max_length (float): total length of edge
            weight (float): particle weight
        Returns:
            None
        """
        self.edge: tuple[int, int] = (start, end)
        self.edge_dist: float = dist
        self.direction: int = direction
        self.max_length: float = max_length
        self.weight: float = weight

    def __repr__(self) -> str:
        """ Returns string representation of particle.
        Args:
            None
        Returns:
            str: formatted particle state
        """
        return (
            f"U:{self.edge[0]:>12d} | "
            f"V:{self.edge[1]:>12d} | "
            f"S:{self.edge_dist:>8.2f} | "
            f"max:{self.max_length:>8.2f} | "
            f"dir:{self.direction:>2d} | "
            f"w:{self.weight:>6.2f}"
        )

    def copy(self) -> "Particle":
        """ Creates a copy of the particle.
        Args:
            None
        Returns:
            Particle: duplicated particle
        """
        return Particle(
            self.edge[0],
            self.edge[1],
            self.edge_dist,
            self.direction,
            self.max_length,
            self.weight,
        )

# ===================== CONSTANTS =====================

DEFAULT_PRTICLE_COUNT = 2000
MIN_PARTICLES = 300

# Constants regarding the re-weighting
PENALTY = 0.5
DISTANCE_DECAY = 200
WEIGHT_FULL_MATCH = 0.1
WEIGHT_PARTIAL_MATCH = 0.05

MATCH_THRESHOLD = 0.85

# =====================================================

class ParticleFilter:
    estonian_letters = "a-zA-ZäöüõÄÖÜÕ\\s"

    def __init__(self, G, particle_count=DEFAULT_PRTICLE_COUNT):
        """ Initialises the particle filter with a graph and particle count.
        Args:
            G (networkx.MultiDiGraph): road network graph
            particle_count (int): number of particles to generate
        Returns:
            None
        """
        self.particles: list[Particle] = []
        self.particle_count = particle_count # particle count might change during resampling
        self.low_weight_counter = 0 # used to trigger resample if filter decays
        self.particle_count_initial = particle_count
        self.G = G

        # graph is projected, transforming is to get the longitude and latitude of the position estimate
        crs = G.graph["crs"]
        self.transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    
        self.edges = list(self.G.edges(keys=False))
        self.edge_length = {
            (start, end): data["length"]
            for start, end, data in self.G.edges(data=True)
        }
        self.edge_geometry = {}

        # It is easier later to check, where particles belong
        self.street_nodes_dict = {}
        self.node_points = {
            n: Point(data["x"], data["y"])
            for n, data in self.G.nodes(data=True)
        }

        # Used to make ure particles are always moving in valid ways
        self.out_edges = {}
        self.in_edges = {}

        self.generate_particles()
        self._get_edge_geometry()
        self._build_street_index()
        self._build_edge_index()

        # Used for match finding
        self.street_names = list(self.street_nodes_dict.keys())

    def _get_edge_geometry(self):
        for u, v, data in self.G.edges(data=True):
            if "geometry" in data:
                self.edge_geometry[(u, v)] = data["geometry"]
            else:
                x1, y1 = self.G.nodes[u]["x"], self.G.nodes[u]["y"]
                x2, y2 = self.G.nodes[v]["x"], self.G.nodes[v]["y"]
                self.edge_geometry[(u, v)] = LineString([(x1, y1), (x2, y2)])

    def _build_street_index(self):
        """ Builds a mapping from street names to nodes. """
        for start, end in self.edges:
            edge_dict = self.G[start][end][0]

            if edge_dict.get("name"):
                names = edge_dict["name"]
                if not isinstance(names, list):
                    names = [names]

                for name in names:
                    name = self.normalise(name)
                    self.street_nodes_dict.setdefault(name, set()).add(start)

    def _build_edge_index(self):
        """ Builds adjacency lists for fast edge traversal. """
        for start, end in self.G.edges():
            self.out_edges.setdefault(start, []).append((start, end))
            self.in_edges.setdefault(end, []).append((start, end))

    def check_weights(self):
        """ Checks if weights are too low and resets particles if needed. """
        if self.low_weight_counter > 5:
            self.low_weight_counter = 0
            self.particles.clear()
            self.generate_particles()

    def generate_particles(self):
        """ Generates random particles across edges. """
        for _ in range(self.particle_count):
            start, end = random.choice(self.edges)
            length = self.edge_length[(start, end)]

            self.particles.append(
                Particle(
                    start,
                    end,
                    random.uniform(0, length),
                    random.choice([1, -1]),
                    length,
                    0.5,
                )
            )

    def update(self, measurement: str):
        """ Updates particle weights based on a text measurement.
        Args:
            measurement (str): observed street name
        Returns:
            bool: True if update applied, False otherwise
        """
        # Check if found text contains valid characters
        if not re.fullmatch(f"[{self.estonian_letters}]+", measurement):
            return False

        match = self.find_matches(measurement)
        if match is None:
            return False

        street_nodes = self.street_nodes_dict.get(match)
        max_weight = 0

        for particle in self.particles:
            u, v = particle.edge
            edge_data = self.G[u][v][0]

            if u in street_nodes and v in street_nodes:
                particle.weight += WEIGHT_FULL_MATCH
            elif u in street_nodes or v in street_nodes:
                particle.weight += WEIGHT_PARTIAL_MATCH
            else:
                # If the particle wasn't directly on or adjacent to the street found, we update based on distance

                particle.weight -= PENALTY

                # Get the Shapely data of the street
                line = self.edge_geometry[(u, v)]

                particle_point = line.interpolate(particle.edge_dist)

                # Calc distances from particle in question to all points on the street found
                distances = [self.node_points[n].distance(particle_point) for n in street_nodes]
                if distances:
                    min_dist = min(distances)
                    # Closer particles will have higher distances
                    particle.weight += PENALTY * (PENALTY ** (min_dist / DISTANCE_DECAY))

            max_weight = max(max_weight, particle.weight)

        # Monitoring the state of the filter
        if max_weight < 0.5:
            self.low_weight_counter += 1
        else:
            self.low_weight_counter -= 1

        return True

    def find_matches(self, measurement: str) -> str:
        """ Finds closest matching street name.
        Args:
            measurement (str): input string
        Returns:
            str: matched street name or None
        """
        measurement = self.normalise(measurement)
        matches = difflib.get_close_matches(measurement, self.street_names, 1, MATCH_THRESHOLD)
        return matches[0] if matches else None

    @staticmethod
    def normalise(s: str) -> str:
        """ normalises string for comparison. """
        return s.lower().strip()

    def predict(self, d_dist):
        """ Moves particles along edges based on distance.
        Args:
            d_dist (float): distance to move
        Returns:
            None
        """
        for particle in self.particles:
            new_dist = particle.edge_dist + d_dist * particle.direction

            # If it is a valid new distance, just accept it
            if 0 < new_dist < particle.max_length:
                particle.edge_dist = new_dist
                continue
            
            """ 
                If we are past our destination node, we take that node, find all edges
                that would allow us to keep moving in the same direction and as a fallback
                option, also edges that would require us to change the direction of travel.
            """
            if new_dist < 0:
                node = particle.edge[0]
                primary_edges = self.in_edges.get(node)
                fallback_edges = self.out_edges.get(node)
            else:
                node = particle.edge[1]
                primary_edges = self.out_edges.get(node)
                fallback_edges = self.in_edges.get(node)

            # Choose edges
            options = primary_edges
            reversed_move = False

            if not options:
                options = fallback_edges
                reversed_move = True

                if not options: # This should never be true since we entered that node from somewhere
                    # Dead-end
                    particle.direction = 0
                    particle.edge_dist = 0 if new_dist < 0 else particle.max_length
                    return
                # Reverse the direction if need be
                particle.direction *= -1

            # Pick next edge
            particle.edge = random.choice(options)
            particle.max_length = self.edge_length[particle.edge]

            # Set new position on edge
            if reversed_move:
                particle.edge_dist = 0 if new_dist < 0 else particle.max_length
            else:
                particle.edge_dist = particle.max_length if new_dist < 0 else 0

    def re_sample(self):
        """ Resamples particles based on weights."""

        # Step 1: extract weights
        weights = np.array([p.weight for p in self.particles], dtype=np.float32)

        # Step 2: prevent negative or zero weights
        weights = np.clip(weights, 0.0, None)
        

        # Step 3: normalise weights
        sum_w = weights.sum()
        if sum_w == 0:
            # All particles have zero weight, make them uniform
            weights = np.ones(len(self.particles)) / len(self.particles)
        else:
            # Normalising the weights
            weights /= np.sum(weights)

        # Step 4: sample indices based on weights
        indices = np.random.choice(len(self.particles), size=len(self.particles), replace=True, p=weights)

        new_particles = []
        edges = set()
        for i in indices:
            p = self.particles[i].copy()
            edges.add(p.edge)
            p.edge_dist = np.random.normal(p.edge_dist)
            p.edge_dist = min(p.max_length, max(0, p.edge_dist))

            p.direction = np.random.choice([1, -1])
            p.weight = 0.5
            new_particles.append(p)
        
        # We find the new count of particles based on the amount of edges our re-sample inhabits
        #particle_count = max(int(len(edges)/len(self.edges) * self.particle_count_initial), MIN_PARTICLES)
        #self.particle_count = particle_count

        # Overwrite old particles
        self.particles[:] = new_particles[:]
        

    def get_position(self, osm_map):
        """ Returns position of best particle in latitude and longitude.
        Args:
            osm_map (OSMMap): map helper for geometry lookup
        Returns:
            tuple: (latitude, longitude)
        """
        best = max(self.particles, key=lambda p: p.weight)

        x, y = osm_map.edge_position(best.edge[0], best.edge[1], best.edge_dist)
        lon, lat = self.transformer.transform(x, y)

        return lon, lat

if __name__ == "__main__":
    start = time.time()
    G = ox.load_graphml("tartu.graphml")
    pf = ParticleFilter(G)
    print(f"set up done in {time.time()-start} sec\n")

    for p in pf.particles[:4]:
        print(p)

    for i in range(10):
        pf.predict(1)
    print()
    for p in pf.particles[:4]:
        print(p)
    

