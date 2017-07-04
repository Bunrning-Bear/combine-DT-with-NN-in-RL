from collections import OrderedDict
def list_to_dic(list):
    dict = {}
    for i in range(0,len(list)):
        dict[str(i)]=list[i]
    return dict

def dic_to_list(dict):
    list = []
    order_list = sorted(dict.items(),key=lambda item:int(item[0]))
    amount = 0
    for item in order_list:
        assert amount == int(item[0]),"missing key %s"%amount
        amount +=1
        list.append(item[1])
    return list