#part 1
num = 100
end = 500
while num < 500:
    num_str = str(num)
    if int(num_str[0]) %2 == 1:
        if int(num_str[1]) % 2 == 1:
            if int(num_str[2]) % 2 == 1:
                print (num)
    num +=1


#part 2
alist = ["1", "4", "0", "6", "9"]
num_list = []

for str_num in alist:
    num_list.append(int(str_num))
print (sorted(num_list))


#part 3
filename = input("enter a file name")
infile = open(filename, "r")

line = infile.readline()
while line != "":
    num_char = len(line)
    num_words = len(line.split( " "))
    print ("number of words: ", num_words)
    print ("number of letters: ", num_char)
    line = infile.readline()





