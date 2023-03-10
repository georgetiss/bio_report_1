#!/usr/bin/env python3
import sys
import random
import json
import numpy
from multiprocessing import pool
import matplotlib.pyplot as plt

from math import atan2, pi
from deap import creator, base, tools, algorithms
from PIL import ImageDraw, Image, ImageChops

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def read_config(PATH, TYPE):
    with open(PATH, "r") as f:
        data = json.load(f)
        return data[TYPE]

TARGET_IMAGE = read_config("config.json", "Image")["TARGET_IMAGE"]
MAX = 255 * 200 * 200
TARGET = Image.open(TARGET_IMAGE)
TARGET.load()  # read image and close the file

def sort_points(x, y):
    centroid = (sum(y) // len(y)), (sum(x) // len(x))
    points = []
    for x, y in zip(x, y):
        angle = atan2(centroid[0] - y, centroid[1] - x)
        points.append(((x, y), (angle) * 180 / pi))

    points.sort(key=lambda y: y[1])

    return points


def make_polygon():
    # 0 <= R|G|B < 256, 30 <= A <= 60, 10 <= x|y < 190
    R, G, B = random.sample(range(0, 235), 3)
    A = random.randint(30, 90)

    x1, y1, x2, y2, x3, y3, x4, y4 = random.sample(range(10, 190), 8)
    p = sort_points([x1, x2, x3, x4], [y1, y2, y3, y4])
    return [(R, G, B, A), (p[0][0]), (p[1][0]), (p[2][0]), (p[3][0])]


def mutate(solution, indpb):
    if random.random() < 0.5:
        # mutate points
        polygon = random.choice(solution)
        coords = [x for point in polygon[1:] for x in point]
        tools.mutGaussian(coords, 0, 10, indpb)
        coords = [max(0, min(int(x), 200)) for x in coords]
        polygon[1:] = list(zip(coords[::2], coords[1::2]))

    else:
        # reorder polygons
        tools.mutShuffleIndexes(solution, indpb)
    return solution,


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])

    return image


def evaluate(solution):
    image = draw(solution) 
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX,


def draw_graph(log):
    fig, ax1 = plt.subplots()
    ax1.plot(log.select("gen"), log.select("median"), color='red', label="Median Fitness")
    fig.tight_layout()

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Median Fitness", color='red')
    for num in ax1.get_yticklabels():
        num.set_color("red")


    return plt


def run(GENERATIONS, POPULATION, MUTATION_PROB,
         TOURNAMENT_SIZE, CROSS_PROB, IND_PROB, POLYGONS, SEED):
    random.seed(SEED)


    toolbox.register("individual", tools.initRepeat, creator.Individual, make_polygon, n=POLYGONS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=IND_PROB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    population = toolbox.population(n=POPULATION)
    hof = tools.HallOfFame(4)
    stats = tools.Statistics(lambda x: x.fitness.values[0])
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("median", numpy.median)
    stats.register("std", numpy.std)
    population, log = algorithms.eaSimple(population, toolbox,
                        cxpb=CROSS_PROB, mutpb=MUTATION_PROB, ngen=GENERATIONS, 
                        stats=stats, halloffame=hof, verbose=True)

    image = draw(population[0])
    image.save("media/solution.png")

    graph = draw_graph(log)
    graph.show()


if __name__ == "__main__":
    params = read_config('config.json', '0')
    run(**params)


