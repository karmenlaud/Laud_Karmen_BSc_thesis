

import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString
from particle_filter import Particle


class OSMMap:
    def __init__(self, G):
        self.G = G

    def edge_position(self, u, v, s):
        """
        Compute (x, y) position along edge (u -> v) at distance s (meters)
        """
        edge_data = self.G[u][v][0]

        if "geometry" in edge_data:
            line = edge_data["geometry"]
        else:
            x1, y1 = self.G.nodes[u]["x"], self.G.nodes[u]["y"]
            x2, y2 = self.G.nodes[v]["x"], self.G.nodes[v]["y"]
            line = LineString([(x1, y1), (x2, y2)])

        s = max(0.0, min(s, line.length))
        point = line.interpolate(s)
        return point.x, point.y

    def render(self, particles: list[Particle], filename="map.png", dpi=300, edge_linewidth=0.6, particle_size=5, cmap_name="plasma"):
        """
        Render the map.
        """
        fig, ax = ox.plot_graph(self.G, show=False, close=False, node_size=0, edge_linewidth=edge_linewidth, bgcolor="white")

        if particles:
            xs, ys, colors = [], [], []

            for p in particles:
                u, v = p.edge
                s = p.edge_dist
                x, y = self.edge_position(u, v, s)
                xs.append(x)
                ys.append(y)
                colors.append(p.weight)  # 0→1

            # Use a colormap
            cmap = plt.cm.get_cmap(cmap_name)
            norm = mcolors.Normalize(vmin=0, vmax=1)
            ax.scatter(xs, ys, c=colors, s=particle_size, cmap=cmap, norm=norm, zorder=5)

        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

