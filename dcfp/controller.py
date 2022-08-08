from dcfp.train import train
from dcfp.comparison import compare
from dcfp.experiment import experiment

class MODELS:
    MLP = 'MLP'

class OPTIMIZERS:
    FBA = 'FBA'
    IFBA = 'IFBA'
    SFBA = 'SFBA' 
    ISFBA = 'ISFBA'
    PFBA = 'PFBA'
    IPFBA = 'IPFBA'
    ParallelSFBA = 'ParallelSFBA'
    ParallelISFBA = 'ParallelISFBA'

class DCFP:
    def __init__(self):
        self.config = None
        self.model = None
        self.optim = None
        self.control_sequences = None
    
    def setup(self, setting_path:str):
        self.config = setting_path
    
    def set_model(self, name:str):
        self.model = name

    def set_optimizer(self, name:str):
        self.optim = name

    def set_control_sequences(self, path:str):
        self.control_sequences = path
    
    # TODO
    # def train(self):
    #     train(data=args.data, 
    #         feature_selection=args.feature_selection,
    #         split_ratio=args.split_ratio,
    #         target=args.target,
    #         disp=args.disp,
    #         args_loss=args.loss,
    #         args_optimizer=args.optimizer,
    #         penalty=args.penalty,
    #         reg_param=args.reg_param,
    #         learning_rate=args.learning_rate,
    #         control_sequence_directory=args.control_sequence_directory,
    #         n_iteration=args.n_iteration,
    #         args_save=args.save,
    #         save_fig=args.not_save_data_fig)
    
    # def experiment(self):
    #     experiment()
    
    def run(self, **kwargs):
        istrain = kwargs['istrain'] if 'istrain' in kwargs.keys() else True
        compare(self.config, istrain=istrain)