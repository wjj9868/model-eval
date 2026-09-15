# @author: ztwz
"""探测 manage_chat_analysis 表结构与可用模型日志分布（只读）。"""
import pymysql

c = pymysql.connect(host="127.0.0.1", port=3306, user="root", password="123456", db="vibee", charset="utf8mb4")
cur = c.cursor()

# manage_chat_analysis 表结构
cur.execute("SHOW CREATE TABLE manage_chat_analysis")
print("[manage_chat_analysis] DDL:")
print(cur.fetchone()[1][:3000])
print()

cur.execute("SELECT COUNT(*) FROM manage_chat_analysis")
print("manage_chat_analysis total:", cur.fetchone()[0])
print()

# response 分布
cur.execute(
    "SELECT status, response='{}' AS empty_resp, COUNT(*) FROM manage_model_log "
    "WHERE task='user_chat_analysis' GROUP BY status, response='{}'"
)
print("response distribution (status, empty, cnt):", cur.fetchall())