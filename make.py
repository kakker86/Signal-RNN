import pandas as pd
import os

def check_file(path):
    if not os.path.exists(path):
        os.makedirs(path)

path1 = os.listdir('./ENG_recorded_DGIST/')
c_len = 0
f_len = 0
d_len = 0
for i, j in enumerate(path1):
    print(j)
    if j[0] == 'b':
        path_c = os.listdir('./ENG_recorded_DGIST/' + j + '/close/')
        path_f = os.listdir('./ENG_recorded_DGIST/' + j + '/far/')

        save_path_c = './data_csv/' + j + '/close/'
        save_path_f = './data_csv/' + j + '/far/'
        check_file(save_path_c)
        check_file(save_path_f)

        bc_count = 0
        for a, b in enumerate(path_c):
            print(b)
            print(b[24:26])
            if b[24:26] == '50':
                c_len = 40
            elif b[24:26] == '10':
                c_len = 180
            elif b[24:26] == '15':
                c_len = 180
            print(f_len)
            data_c_xlsx = pd.read_excel('./ENG_recorded_DGIST/' + j + '/close/' + b)
            data_c_xlsx = data_c_xlsx.drop(['Time'], axis=1)
            data_c_xlsx[1] = 0
            dc = data_c_xlsx.iloc[10+bc_count:20+bc_count+(c_len*3)]
            print(dc)
            bc_count = bc_count + 31
            dc.to_csv(save_path_c + b[:-4] + 'csv', index=False, header=False)

        bf_count = 0
        for c, d in enumerate(path_f):
            print(d)
            print(d[22:24])
            if d[22:24] == '50':
                f_len = 40
                flabel_v = 1
            elif d[22:24] == '10':
                f_len = 180
                flabel_v = 2
            elif d[22:24] == '15':
                f_len = 180
                flabel_v = 3

            print(f_len)

            data_f_xlsx = pd.read_excel('./ENG_recorded_DGIST/' + j + '/far/' + d)
            data_f_xlsx = data_f_xlsx.drop(['Time'], axis=1)
            data_f_xlsx[1] = 0
            data_f_xlsx[1].iloc[20+bf_count:20+bf_count+f_len] = flabel_v
            df = data_f_xlsx.iloc[10+bf_count:20+bf_count+(f_len*3)]
            print(df)
            bf_count = bf_count + 31
            df.to_csv(save_path_f + d[:-4] + 'csv', index=False, header=False)

    elif j[0] == 'd':
        path2 = os.listdir('./ENG_recorded_DGIST/' + j)

        save_path = './data_csv/' + j + '/'
        check_file(save_path)

        d_count = 0
        d_start = 20


        for n, m in enumerate(path2):
            print(m)
            print(m[10:12])
            if m[10:12] == '50':
                d_len = 40
                dlabel_v = 4
            elif m[10:12] == '10':
                d_len = 180
                dlabel_v = 5
            elif m[10:12] == '20':
                d_len = 180
                dlabel_v = 6
            print(d_len)
            data_xlsx = pd.read_excel('./ENG_recorded_DGIST/' + j + '/' + m)
            data_xlsx = data_xlsx.drop(['Time'], axis=1)
            data_xlsx[1] = 0
            data_xlsx[1].iloc[20+d_count:20+d_count+d_len] = dlabel_v
            db = data_xlsx.iloc[10+d_count:20+d_count+(d_len*3)]
            print(db)
            d_count = d_count + 31
            db.to_csv(save_path + m[:-4] + 'csv', index=False, header=False)