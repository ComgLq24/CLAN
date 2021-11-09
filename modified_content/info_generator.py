import csv
import glob


def main():
    # open csv file and read new information
    with open("new_info.csv") as fi:
        text = csv.reader(fi)
        colors = ['#000000', ]
        names = ['background', ]
        for item in text:
            names.append(item[0])
            colors.append(item[1])
    # output colormap
    color_map = {}
    color_name = {}
    for index in range(0, len(colors)):
        color_map[colors[index]] = index
        color_name[colors[index]] = names[index]
    print(color_map)
    print(color_name)

    # generate 'train.txt' and 'val.txt'
    # with open('../dataset/gta5_list/train.txt', 'w') as fo:
    #     for path in glob.glob('../data/GTA5/images/*.*'):
    #         fo.write('{0}\n'.format(path.split('/')[-1]))
    #
    # with open('../dataset/cityscapes_list/train.txt', 'w') as fo:
    #     for path in glob.glob('../data/Cityscapes/leftImg8bit/train/*.*'):
    #         fo.write('{0}\n'.format(path.split('/')[-1]))

    with open('../dataset/cityscapes_list/val.txt', 'w') as fo:
        for path in glob.glob('../data/Cityscapes/leftImg8bit/val/*.*'):
            fo.write('{0}\n'.format(path.split('/')[-1]))


if __name__ == '__main__':
    main()
