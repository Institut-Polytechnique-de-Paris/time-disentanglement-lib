class XGenTrainingConfig:
    def __new__(self,
                output_dir='my_model',
                num_epochs=50,
                learning_rate=1e-3,
                per_device_train_batch_size=100,
                per_device_eval_batch_size=100,
                train_dataloader_num_workers=0,
                eval_dataloader_num_workers=0,
                steps_saving='auto',
                optimizer_cls="Adamax",
                optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
                scheduler_cls="ReduceLROnPlateau",
                scheduler_params={"patience": 5, "factor": 0.5}):
        
        self.training_config = {
            "seq_length": 512,
            "batch_size" : 64,
            "embedding_dim" : 10,
            "value_features" : 1,
            "key_features" : 1,
            "discriminator_lr" : 0.00005,
            "generator_lr" : 0.00005,
            "num_epochs" : 10,
            "blocks_to_add" : 2,
            "timestamp" : 5,
            "ml" : True,
            "fade_in" : True,
            "sa" : True,
            "save" : True,
            "gpu" : False,
            "name": "BaseModel",
            "path": "Models/M4/",
        }
        
        return self.training_config