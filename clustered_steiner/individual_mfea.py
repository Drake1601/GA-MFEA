class individual_mfea():
    def __init__(self,steiner_vertexs,clusters):
        self.steiner_vertexs = steiner_vertexs
        self.clusters = clusters

    def set_fitness(self,fitness):
        self.fitness = fitness

    def update_ST(self,ST):
        self.ST = ST

