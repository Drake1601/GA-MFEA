import configparser
from init import init_population,init_encode,init_population_mfea
from tqdm import tqdm
import random
import utils
from crossover import CrossOver
import numpy as np

random.seed(1)
config_path = "config/config.ini"
config = configparser.ConfigParser()
config.read(config_path)

POPULATION_SIZE = int(config["general"]["population_size"])
NUM_GENERATION = int(config["general"]["num_generation"])
mutation_rate = float(config["general"]["mutation_rate"])
crossover_rate = float(config["general"]["crossover_rate"])

graph = []
clusters = []
vertexs = []

graph_path1 = "graph.txt"
graph1, clusters1 = utils.load_data(graph_path1)
vertexs1 = set(range(graph1.shape[0]))

graph.append(graph1)
clusters.append(clusters1)
vertexs.append(vertexs1)


graph_path2 = "10a280.txt"
graph2, clusters2 = utils.load_data(graph_path2)
vertexs2 = set(range(graph2.shape[0]))

graph.append(graph2)
clusters.append(clusters2)
vertexs.append(vertexs2)

# init_code = init_population_mfea(
#     population_size = POPULATION_SIZE,
#     chromosomes_size = max(len(utils.getSteinerVertexs(vertexs1,clusters1)),
#                            len(utils.getSteinerVertexs(vertexs2,clusters2)))
#     )
#
# rank = []
# rank1 = utils.calculate_rank(init_code,graph1,vertexs1,clusters1)
# rank2 = utils.calculate_rank(init_code,graph2,vertexs2,clusters2)
# rank.append(rank1)
# rank.append(rank2)
#
# best = init_code[0]
# best_dis = 100000
# init_sf = utils.calculate_skill_factor(rank)
# population = init_code
#
# print(int((0.99-0.2)/0.08))
# for i in range(10):
#     random.seed(1)
#     np.random.seed(1)
#     new_population = CrossOver.crossover(init_code,init_sf)
#     print(len(new_population))
#     print(len(init_sf))
leng1 = len(utils.getSteinerVertexs(vertexs1,clusters1))
leng2 = len(utils.getSteinerVertexs(vertexs2,clusters2))
test = init_encode(max(leng1,leng2))
# print(test[0])

decode = utils.decode(test,10,utils.getSteinerVertexs(vertexs1,clusters1))
# print(decode.cluster_index)

init_code = init_population_mfea(10,max(leng1,leng2))
rank = []
rank1 = utils.calculate_rank(init_code,graph1,vertexs1,clusters1)
rank2 = utils.calculate_rank(init_code,graph2,vertexs2,clusters2)
rank.append(rank1)
rank.append(rank2)

init_sf = utils.calculate_skill_factor(rank)
c1,c2 = CrossOver.crossover_swap(init_code[1],init_code[2])
c3 = CrossOver.mutate(init_code[0])
print(c3)