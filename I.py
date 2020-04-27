import sys
import time
MyFeelingsForYou=2 #when i see u the first time
def myLife():
    return 100

print("CountMyFeelingsForYou:")
input()
while True:
    MyFeelingsForYou += MyFeelingsForYou
    print(MyFeelingsForYou)
    sys.stdout.write("\033[F")
    if len(str(MyFeelingsForYou))>myLife():
        break 
    time.sleep(.05)
print("I death....")