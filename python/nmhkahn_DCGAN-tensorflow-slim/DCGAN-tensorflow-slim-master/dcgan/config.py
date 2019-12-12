class Config(object):
    def __init__(self, **kwargs):
        # configuration for building the network
        self.y_dim = kwargs.get("y_dim", 5)
        self.z_dim = kwargs.get("z_dim", 100)

        self.batch_size = kwargs.get("batch_size", 256)
        self.lr = kwargs.get("lr", 0.0002)
        self.beta1 = kwargs.get("beta1", 0.5) # recoomand value in dcgan paper

        # configuration for the supervisor
        self.logdir = kwargs.get("logdir", "./log")
        self.sampledir = kwargs.get("sampledir", "./example")

        self.max_steps = kwargs.get("max_steps", 100000)
        self.sample_every_n_steps = kwargs.get("sample_every_n_steps", 1000)
        self.summary_every_n_steps = kwargs.get("summary_every_n_steps", 10)
        self.save_model_secs = kwargs.get("save_model_secs", 1200)
        self.checkpoint_basename = kwargs.get("checkpoint_basename", "dcgan")

        # configuration for the dataset queue
        self.filename_queue = kwargs["filename_queue"]
        self.min_after_dequeue = kwargs.get("min_after_dequeue", 5000)
        self.num_threads = kwargs.get("num_threads", 4)
