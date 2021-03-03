class User:
    def __init__(self,rank, name, mode, corr, mmc,, stake, day_1=None,month_3=None,year_1 =None):
        self.rank = rank
        self.name = name
        self.mode = mode
        self.corr = corr
        self.mmc = mmc
        self.stake = stake
        self.day_1 = day_1
        self.month_3 = month_3
        self.year_1 = year_1

def main():
    with open("scores.txt",'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        record =  lines[:4]
        print(len(record))
        rank = record[0]
        name = record[1]
        mode = record[2]
        stats = record[3].split('\t')
    
        corr = stats[0]
        mmc = stats[1]
        try:
            stake = stats[2]
        except:
            state =None
        if len(stats)==4:
            day_1 = stats[3]
            month_3 = None
            year_1 = None
        elif len(stats)==5:
            day_1 = stats[3]
            month_3 = stats[4]
            year_1 = None
        elif len(stats)==6:
            day_1 = stats[3]
            month_3 = stats[4]
            year_1 = stats[5]
        
        my_user = User(rank, name, mode, corr, mmc,  stake, day_1,month_3,year_1)
        print(type(my_user))
        print(record)



# I think it makes the most sense to cast this in to a relational database for some basic quries. To do that I need to cast it as a .csv file

        
main()