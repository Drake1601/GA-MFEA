import random
from individual import individual
import utils


graph_path1 = "graph.txt"
graph1, clusters1 = utils.load_data(graph_path1)
vertexs1 = set(range(graph1.shape[0]))

graph_path2 = "10a280.txt"
graph2, clusters2 = utils.load_data(graph_path2)
vertexs2 = set(range(graph2.shape[0]))

class CrossOver():
    def __init__(self) -> None:
        pass
    
    @classmethod
    def check_steiner_vertexs(cls, individual_1, individual_2):
        assert len(individual_1.steiner_vertexs) == len(individual_2.steiner_vertexs)
        length = len(individual_1.steiner_vertexs)
        
        for i in range(length):
            assert individual_1.steiner_vertexs[i] == individual_2.steiner_vertexs[i]
        
    @classmethod
    def crossover(cls, parent_1, parent_2):
        cls.check_steiner_vertexs(parent_1, parent_2)
        length = len(parent_1.gene)
        crossover_index = random.randint(1, length-1)
        
        #gene_1 = parent_1.gene[0:crossover_index] + parent_2.gene[crossover_index:]
        cluster_index_1 = parent_1.cluster_index[0:crossover_index] + parent_2.cluster_index[crossover_index:]
        steiner_vertexs_1 = parent_1.steiner_vertexs
        
        child_1 = individual(
            steiner_vertexs=steiner_vertexs_1,
           # gene=gene_1,
            cluster_index=cluster_index_1
        )
        
        #gene_2 = parent_2.gene[0:crossover_index] + parent_1.gene[crossover_index:]
        cluster_index_2 = parent_2.cluster_index[0:crossover_index] + parent_1.cluster_index[crossover_index:]
        steiner_vertexs_2 = parent_1.steiner_vertexs
        
        child_2 = individual(
            steiner_vertexs=steiner_vertexs_2,
        #    gene=gene_2,
            cluster_index=cluster_index_2
        )
        
        return child_1, child_2
    
    @classmethod
    def mutate_gene(cls, indi):
        new_gen = list(indi.gene)
        length = len(indi.gene)
        
        index = random.randint(0, length-1)
        new_gen[index] = 1 - new_gen[index]
        indi.gene = tuple(new_gen)
        return indi

    @classmethod
    def crossover_swap(cls,pa1,pa2):
        index1 = random.randint(0,len(pa1)-20)
        index2 = random.randint(index1,len(pa1)-1)
        child1 = pa1[0:index1]+pa2[index1:index2]+pa1[index2:]
        child2 = pa2[0:index1]+pa1[index1:index2]+pa2[index2:]

        return child1,child2

    @classmethod
    def crossover_sbx(cls,parent1,parent2):
        u = random.random()
        t = random.randint(2,11)
        if u <= 0.5:
            beta = (2*u)**(1/t)
        else:
            beta = (1/(2*(1-u)))**(1/t)

        child1 = []
        child2 = []
        for i in range(len(parent1)):
            c1 = 0.5*((1+beta)*parent1[i]+(1-beta)*parent2[i])
            c2 = 0.5*((1-beta)*parent1[i]+(1+beta)*parent2[i])


            child1.append(abs(c1) if abs(c1)<1 else 1.0/abs(c1))
            child2.append(abs(c2) if abs(c2)<1 else 1.0/abs(c2))

        return child1,child2

    @classmethod
    def mutate(cls,indi_encode):
        tmp = []
        leng = len(indi_encode)
        index1 = random.randint(0,leng-10)
        index2 = random.randint(index1,leng-1)
        for i in range(leng):
            if indi_encode[i] > 1:
                print('error > 1')
            if indi_encode[i] < 0:
                print('error < 0')
            tmp.append(1-indi_encode[i])

        new_encode = indi_encode[:index1]+tmp[index1:index2]+indi_encode[index2:]
        return  new_encode

    # @classmethod
    # def crossover_mfea(cls,pa1,pa2):
    #     r = random.random()
    #     if r

    @classmethod
    def crossover(cls,population,init_sf):
        new_population = []
        r = random.random()
        POPULATION_SIZE = len(population)
        for i in range(0, POPULATION_SIZE - 1, 2):
            s1 = utils.get_skill_factor(init_sf, population[i])
            s2 = utils.get_skill_factor(init_sf, population[i + 1])

            if r < 0.3 or s1 == s2:
                c1, c2 = CrossOver.crossover_swap(population[i], population[i + 1])
                new_population.append(c1)
                new_population.append(c2)
                u = random.random()
                if u < 0.5:
                    init_sf.append((c1, s1))
                    init_sf.append((c2, s1))
                else:
                    init_sf.append((c1, s2))
                    init_sf.append((c2, s2))
            else:
                c1 = CrossOver.mutate(population[i])
                c2 = CrossOver.mutate(population[i + 1])
                new_population.append(c1)
                new_population.append(c2)
                init_sf.append((c1, s1))
                init_sf.append((c2, s2))
        return new_population

    @classmethod
    def selection(cls,population,init_sf,clusters,vertexs,graph):
        fitness = []
        task1 = []
        task2 = []
        fitness.append(task1)
        fitness.append(task2)
        # random.seed(1)
        for indi_code in population:
            for sf in init_sf:
                # random.seed(1)
                if indi_code == sf[0]:
                    #   count.append(indi_code)

                    index = int(sf[1]) - 1
                    #
                    indi = utils.decode(indi_code, len(clusters[index]),
                                        utils.getSteinerVertexs(vertexs[index], clusters[index]))
                    # print(indi.cluster_index)
                    # print(len(indi.cluster_index))
                    fitness[index].append((indi_code, utils.calculate_fitness(indi, clusters[index],
                                                                              graph[index])))

        fitness[0] = sorted(fitness[0], key=lambda x: x[1], reverse=False)
        fitness[1] = sorted(fitness[1], key=lambda x: x[1], reverse=False)

        # print(fitness[0][0][1])

        best_population = []
        best_init_sf = []

        for i in range(50):
            best_population.append(fitness[0][i][0])
            best_population.append(fitness[1][i][0])
            best_init_sf.append((fitness[0][i][0], 1))
            best_init_sf.append((fitness[1][i][0], 2))

        # tmp1 = fitness[0][30:]
        # tmp2 = fitness[1][30:]
        # random.shuffle(tmp1)
        # random.shuffle(tmp2)
        # for i in range(20):
        #   best_population.append(tmp1[i][0])
        #   best_population.append(tmp2[i][0])
        #   best_init_sf.append((tmp1[i][0], 1))
        #   best_init_sf.append((tmp2[i][0], 2))
        # random.shuffle(best_population)
        # random.shuffle(best_init_sf)

        # random.shuffle(best_population)
        # random.shuffle(best_init_sf)

        return best_population,best_init_sf,fitness[0][0][1],fitness[0][0][0]
