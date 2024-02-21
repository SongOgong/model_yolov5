import glob

input_dir='C:\\Users\\syjin\\Desktop\\object\\yolov5\\laundry\\labels\\'
f = open('./train_list.txt', 'w')
input_file=glob.glob(input_dir+'*.txt')

for file in input_file:
    f.write(file[:41]+'images\\'+file[48:-4]+'.jpg \n')
f.close()


f = open('./val_list.txt', 'w')

for file in input_file:
    f.write(file[:41]+'images\\'+file[48:-4]+'.jpg \n')
f.close()
