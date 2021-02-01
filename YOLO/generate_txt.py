import os

train_file = open("train.txt", "w")
test_file = open("test.txt", "w")

current_path = os.getcwd()
print(current_path)

for i in os.listdir(current_path):
    if os.path.isdir(i):
        dir = current_path + "\\" + i
        counter = 0
        for j in os.listdir(dir):
            # print(j)
            if counter < 300:
                if j[-3:] == 'jpg':
                    # print(j[-3:] =='jpg' or 'JPG')
                    train_file.write(dir + '/' + j + '#')
                    # print("train"+dir+'/'+j)
                    # print("wrrrr")
                    train_file.close
                else:
                    train_file.write(dir + '/' + j + "\n")
                    train_file.close
                    counter = counter + 1
                    # print("train"+dir+'/'+j)
                    # print(counter)

            else:
                if j[-3:] == 'jpg':
                    test_file.write(dir + '/' + j + '#')
                    test_file.close
                else:
                    test_file.write(dir + '/' + j + "\n")
                    test_file.close
                    counter = counter + 1
                