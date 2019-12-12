

class LearnerLog:
    '''
    A log is composed of:
        - static key,value pairs (for example: hyper parameters of the experiment)
        - set of key,value pairs at each iteration

    Typical use is:
        log.add_static_value("learning_rate",0.01)

        for t in range(T):
            perf=evaluate_model()
            log.new_iteration()
            log.add_dynamic_value("perf",perf)
            log.add_dynamic_value("iteration",t)
    '''
    def __init__(self):
        self.svar={}
        self.dvar=[]
        self.t=-1

    def add_static_value(self,key,value):
        self.svar[key]=value

    def new_iteration(self):
        self.t=self.t+1
        self.dvar.append({})

    def add_dynamic_value(self,key,value):
        self.dvar[self.t][key]=value

    def get_last_dynamic_value(self,key):
        return self.dvar[self.t][key]