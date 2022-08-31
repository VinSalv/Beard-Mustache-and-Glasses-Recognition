class TrainImage:
    def __init__(self, file_name, folder):
        self.name = file_name.rpartition('.')[0]
        self.ext = file_name.rpartition('.')[-1]
        self.path = folder + '/' + file_name

    def is_equal(self, file_name):
        return self.name == file_name.rpartition('.')[0] \
               and \
               self.ext == file_name.rpartition('.')[-1]
