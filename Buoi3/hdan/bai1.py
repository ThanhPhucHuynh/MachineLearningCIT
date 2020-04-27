import pandas as pd

dt = pd.read_csv("play_tennis.csv")

print(dt)

# find outLook Play = yes
dtO = dt.outlook[dt.play=="Yes"]
print(dtO)
#xx
P1_1 = dtO.value_counts()
P1_1 = P1_1/dtO.count()
print(P1_1)


# find outLook Play = no
dtO = dt.outlook[dt.play=="No"]
print(dtO)
#xx
P1_2 = dtO.value_counts()
P1_2 = P1_2/dtO.count()
print(P1_2)



# find temp Play = yes
dtO = dt.temp[dt.play=="Yes"]
print(dtO)
#xx
P2_1 = dtO.value_counts()
P2_1 = P2_1/dtO.count()
print(P2_1)


# find temp Play = no
dtO = dt.temp[dt.play=="No"]
print(dtO)
#xx
P2_2 = dtO.value_counts()
P2_2 = P2_2/dtO.count()
print(P2_2)



# find humidity Play = yes
dtO = dt.humidity[dt.play=="Yes"]
print(dtO)
#xx
P3_1 = dtO.value_counts()
P3_1 = P3_1/dtO.count()
print(P3_1)


# find humidity Play = no
dtO = dt.humidity[dt.play=="No"]
print(dtO)
#xx
P3_2 = dtO.value_counts()
P3_2 = P3_2/dtO.count()
print(P3_2)


# find wind Play = yes
dtO = dt.wind[dt.play=="Yes"]
print(dtO)
#xx
P4_1 = dtO.value_counts()
P4_1 = P4_1/dtO.count()
print(P4_1)


# find wind Play = no
dtO = dt.wind[dt.play=="No"]
print(dtO)
#xx
P4_2 = dtO.value_counts()
P4_2 = P4_2/dtO.count()
print(P4_2)



P = dt.play.value_counts()/dt.play.count()

print(P)

#(rainy,cool,high,F)

# print(P4_1[0],P1_1.Rain)
P_yes = P1_1.Rain*P2_1.Cool*P3_1.High*P4_1[0]*P.Yes

# print(P_yes)
P_no = P1_2.Rain*P2_2.Cool*P3_2.High*P4_2[0]*P.No
# print(P_no)

PY = P_yes/(P_no+P_yes)
PN = P_no/(P_no+P_yes)

print(PY, PN)


