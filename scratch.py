
#problem 2
import string

number = input(" enter a three digit number")
snum = str(number)
print(snum[::-1])




operation = input("enter an option: \n1. +\n2. -\n3. *\n4. /\n5. %\n")
a = input ("enter a number")
a = int (a)
b = input ("enter a number")
b = int (b)

if operation =="1":
    print (a+b)
elif operation =="2":
    print (a-b)
elif operation =="3":
    print (a*b)
elif operation =="4":
    print (a/b)
elif operation =="5":
    print (a%b)
else:
    print("invalid operation")


#problem 3


sentence =  input( "enter a sentence")
sentence = sentence.split(" ")


print ("the number of words are ", len(sentence))

letters = 0
digits = 0

for word in sentence:
    for char in word:
        if char.isdigit():
            digits+=1
        elif char.isalpha():
            letters+=1

print ("letters: ", letters)
print ("digits: ", digits)


