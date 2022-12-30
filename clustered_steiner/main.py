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

rand = random.Random(1)
np.random.seed(1)


# population = init_population(
#     vertexs=vertexs1,
#     clusters=clusters1,
#     population_size=POPULATION_SIZE
# )
#
# for ele in population:
#     fitness = utils.calculate_fitness(
#         individual=ele,
#         clusters=clusters1,
#         graph=graph1
#     )
#     ele.set_fitness(fitness)

init_code = init_population_mfea(
    population_size = POPULATION_SIZE,
    chromosomes_size = max(len(utils.getSteinerVertexs(vertexs1,clusters1)),
                           len(utils.getSteinerVertexs(vertexs2,clusters2)))
    )

rank = []
rank1 = utils.calculate_rank(init_code,graph1,vertexs1,clusters1)
rank2 = utils.calculate_rank(init_code,graph2,vertexs2,clusters2)
rank.append(rank1)
rank.append(rank2)

best = init_code[0]
best_dis = 100000
init_sf = utils.calculate_skill_factor(rank)
population = init_code
# test = utils.find_MST(graph1,vertexs1).tolist()
# # print(test)
# c1,c2 = CrossOver.crossover_sbx(init_code[1],init_code[2])
# indi_test1 = utils.decode(init_code[5],10,utils.getSteinerVertexs(vertexs1,clusters1))
# indi_test2 = utils.decode(init_code[10],10,utils.getSteinerVertexs(vertexs1,clusters1))
# for i in range(10):
#     random.seed(1)
#     print(utils.calculate_fitness(indi_test1,clusters1,graph1))
#     print(utils.calculate_fitness(indi_test2,clusters1,graph1))
# random.seed(1)
# np.random.seed(1)
############################################################
temp = []
_tqdm = tqdm(range(NUM_GENERATION))
for _ in _tqdm:
    rand = random.seed(1)
    # tmp = utils.decode(init_code[0], 10, utils.getSteinerVertexs(vertexs1, clusters1))
    # print('', utils.calculate_fitness(tmp, clusters1, graph1))
    # if best == population[0]:
    #     print('True')
    #print(' ')
    #
    new_population = [] # encode for new population
    r = random.random()
    # tmp = utils.decode(init_code[0], 10, utils.getSteinerVertexs(vertexs1, clusters1))
    # print('', utils.calculate_fitness(tmp, clusters1, graph1))
    #
    new_population = CrossOver.crossover(population,init_sf)

    # tmp = utils.decode(init_code[0], 10, utils.getSteinerVertexs(vertexs1, clusters1))
    # print('', utils.calculate_fitness(tmp, clusters1, graph1))

    # fitness = []
    # # fitness[0] = []
    # # fitness[1] = []
    # task1 = []
    # task2 = []
    # fitness.append(task1)
    # fitness.append(task2)
    population = population + new_population

    # for indi_code in population:
    #     for sf in init_sf:
    #         if indi_code == sf[0]:
    #          #   count.append(indi_code)
    #
    #             index = int(sf[1])-1
    #             #
    #             indi = utils.decode(indi_code,len(clusters[index]),
    #                                 utils.getSteinerVertexs(vertexs[index],clusters[index]))
    #             # print(indi.cluster_index)
    #             # print(len(indi.cluster_index))
    #             fitness[index].append((indi_code,utils.calculate_fitness(indi,clusters[index],
    #                                                                 graph[index])))
    # best_task1 = []
    # # print(len(fitness[0]),len(fitness[1]))
    # fitness[0] = sorted(fitness[0],key = lambda x:x[1],reverse=False)
    # fitness[1] = sorted(fitness[1],key = lambda x:x[1],reverse=False)
    #
    # best_population = []
    # best_init_sf = []
    #
    # tmp = utils.decode(init_code[0], 10, utils.getSteinerVertexs(vertexs1, clusters1))
    # print('', utils.calculate_fitness(tmp, clusters1, graph1))
    #
    # #
    # # if fitness[0][0][0] == best:
    # #     print('best k doi')
    # # else :
    # #     tmp = utils.decode(best,10,utils.getSteinerVertexs(vertexs1,clusters1))
    # #     print(utils.calculate_fitness(tmp,clusters1,graph1))
    # best = fitness[0][0][0]
    # #print(utils.get_skill_factor(init_sf,best))
    # # tmp = utils.decode(best, 10, utils.getSteinerVertexs(vertexs1, clusters1))
    # # print('',utils.calculate_fitness(tmp, clusters1, graph1))
    # # print(utils.calculate_fitness(tmp, clusters1, graph1))
    # # print(utils.calculate_fitness(tmp, clusters1, graph1))
    # # print(utils.calculate_fitness(tmp, clusters1, graph1))
    # best_dis = fitness[0][0][1]
    # print(best_dis)
    # #
    # #temp.append(best)
    # # print(len(best))
    # #print('best the he hien tai:',task1[0][1])
    # # print(utils.get_skill_factor(init_sf,task1[0][0]))
    # #print(task2[0][1])
    # for i in range(50):
    #     best_population.append(fitness[0][i][0])
    #     best_population.append(fitness[1][i][0])
    #     best_init_sf.append((fitness[0][i][0],1))
    #     best_init_sf.append((fitness[1][i][0],2))
    # random.shuffle(best_population)
    # random.shuffle(best_init_sf)
    # population = []
    # init_sf = []
    # #print(len(population))
    # population = best_population
    # init_sf = best_init_sf
    # tmp = utils.decode(init_code[0], 10, utils.getSteinerVertexs(vertexs1, clusters1))
    # print('',utils.calculate_fitness(tmp, clusters1, graph1))
    # print(utils.get_skill_factor(init_sf,best))
    # print('check',utils.check(population,best))
    t_population,t_init_sf,best_dis,best = CrossOver.selection(population,init_sf,clusters,vertexs,graph)
    # print('check', utils.check(population,best))
    # tmp = utils.decode(init_code[0], 10, utils.getSteinerVertexs(vertexs1, clusters1))
    # print('',utils.calculate_fitness(tmp, clusters1, graph1))
    #print(t_population[0])
    population = t_population
    random.shuffle(population)
    # #print(population[0])
    init_sf = t_init_sf
    print('best_dis',best_dis)


############################################################
# for i in range(10):
#     random.seed(1)
#     np.random.seed(1)
#     for j in range(len(temp)):
#         tmp = utils.decode(temp[j],10,utils.getSteinerVertexs(vertexs1,clusters1))
#         print(i,j,utils.calculate_fitness(tmp,clusters1,graph1))
    #print(best_init_sf[len(best_init_sf)-1])

    # print(len(init_sf))
    # print(len(best_population))
    # print(len(best_init_sf))
    #print(best_init_sf[0][0],best_init_sf[0][1])

    # tmp = 0
    # tmp2 = 0
    # for i in population:
    #     for j in init_sf:
    #         if i==j[0] and j[1] == 1:
    #             tmp +=1
    #         if i==j[0] and j[1] == 2:
    #             tmp2 +=1
    # print(tmp,tmp2)






# _tqdm = tqdm(range(NUM_GENERATION))
# for _ in _tqdm:
#     news_population = []
#     mutation_set = set(random.sample(population, k=int(mutation_rate*POPULATION_SIZE)))
#
#     crossover_set = list(set(population) - mutation_set)
#     mutation_set = list(mutation_set)
#
#     for i in range(int(len(crossover_set)/2)):
#         parents = random.sample(crossover_set, k=2)
#
#         child_1, child_2 = CrossOver.crossover(
#             *parents
#         )
#         news_population.append(child_1)
#         news_population.append(child_2)
#
#     for indi in mutation_set:
#         CrossOver.mutate_gene(indi)
#     utils.update_fitness(news_population, graph=graph1, clusters=clusters1)
#
#     population += news_population
#
#     population.sort()
#     population = population[0:POPULATION_SIZE]
#
#     _tqdm.set_postfix({
#         "bestfound":population[0].fitness
#     })