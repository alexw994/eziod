import eziod
import os

class test_dataset(eziod.IOD):
    def __init__(self):
        self.data_dir = 'test_data'
        self.cache_file = os.path.join(self.data_dir, 'test.cache')
        super(test_dataset, self).__init__(self.cache_file, self.data_dir, download_url=None)

    def init_images(self, data_dir):
        # 图像数据
        self.download_data(r'http://172.29.231.45:8001/f/229bda4021e04f3ea865/?dl=1', os.path.join(data_dir, 'images.zip'))
        self.unzip(os.path.join(data_dir, 'images.zip'))

        # anno文件
        self.download_data(r'http://172.29.231.45:8001/f/3f5d2073820d4713a96b/?dl=1', os.path.join(data_dir, 'anno.txt'))

    def parse_anno(self, data_dir):
        img_infos = []
        with open(os.path.join(data_dir, 'anno.txt'), 'r') as f:
            for l in f.readlines():
                img_path, width, height, faces = l.strip().split(' ')
                info = eziod.ImgInfo(img_path = img_path,
                                     width = width,
                                     height = height)
                if not isinstance(faces, list):
                    faces = [faces]

                for face in faces:
                    corrs = list(map(int, face.split(',')))
                    box = corrs[0:4]
                    landmark = corrs[4:]
                    obj = eziod.Obj(class_name = 'nomal_face',
                                    class_index='0',
                                    box = box,
                                    landmark = landmark)
                    info.set_objs(obj)

                img_infos.append(info)
        return img_infos


if __name__ == '__main__':
    test_dataset()


