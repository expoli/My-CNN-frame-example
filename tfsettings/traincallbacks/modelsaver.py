class ModelSaver:
    def __init__(self, ):
        self.model_save_path = ''

    def set_model_path(self, save_path):
        self.model_save_path = save_path
        return self.model_save_path

    def save(self, model):
        print('[INFO] saving model to folder.....')
        model.save(self.model_save_path)
        return self.model_save_path
