import argparse

from detection.FireDetection import FireDetectioner
from tfsettings.gpu import InitGpu

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to test dataset')
ap.add_argument('-m', '--model', required=True, help='saved model file path')
ap.add_argument('-c', '--categories', required=True, help='categories')
ap.add_argument('-gui', '--gui_flag', required=True, help='the gui out flag')
args = vars(ap.parse_args())


def main():
    InitGpu.InitGpu().init()
    model_path = args['model']
    dataset_path = args['dataset']
    categories = args['categories']
    video_path = dataset_path + '/' + categories
    gui_flag = args['gui_flag']
    firedetectioner = FireDetectioner(modelPath=model_path, video_path=video_path, gui_flag=gui_flag)
    firedetectioner.detection()


if __name__ == '__main__':
    main()
