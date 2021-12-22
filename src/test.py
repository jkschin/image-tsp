import src.main.generate_data as gd

if __name__ == "__main__":
    n = 20
    delta = 10
    size = (512, 512, 3)
    coords = gd.generate_coordinates(n, delta, size)
    subset = random.sample(coords, 10)
    dists = hk.generate_manhattan_distances(subset)
    tsp_sol = generate_tsp_sol(subset)
    routes = gortools.main(dists)
