import os
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

# MySQL连接参数
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'aks',
    'database': 'HS300'
}

# CSV 文件所在的目录
csv_directory = '../newstore/'

# 获取目录中所有的 CSV 文件
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# 创建数据库连接
# conn = mysql.connector.connect(**db_config)
conn = create_engine('mysql+pymysql://root:aks@localhost:3306/HS300?charset=utf8')
# cursor = conn.cursor()

# 批量导入 CSV 文件
for csv_file in csv_files:
    print(csv_file)
    table_name = csv_file[:-4]  # 假设文件名与表格名相同（去掉.csv后缀）

    # 读取 CSV 文件
    df = pd.read_csv(os.path.join(csv_directory, csv_file))

    # 将数据导入到 MySQL 数据库中
    df.to_sql(name='hs'+table_name, con=conn, if_exists='replace', index=False)

    print(f'Table {table_name} imported successfully.')

# 关闭数据库连接
# cursor.close()
# conn.close()