
import pandas as pd
tmp = pd.read_excel("CrCoNi.xlsx",sheet_name="SUM")
print(tmp)

#print(tmp)
for row in range(3,7):
#for row in range(len(tmp)):計算有幾個row
    #print(row)
    #print(tmp.iloc[row],row)
    x = int(tmp.iloc[row]['x'])
    y = int(tmp.iloc[row]['y'])
    z = int(tmp.iloc[row]['z'])
    sheet = pd.read_excel("CrCoNi.xlsx",header=None,sheet_name=f"{x}{y}{z}")
    #print(sheet)
    #tmp.iloc[row]["UTS"]=sheet[1].max()
    tmp.loc[row,"UTS"]=sheet[1].max()
print(tmp)
tmp.to_excel("CrCoNi.xlsx",sheet_name="SUM")


