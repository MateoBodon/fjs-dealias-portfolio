from utils.credentials import wrds_creds
import psycopg2

HOST = "wrds-pgdata.wharton.upenn.edu"
PORT = 9737
DB   = "wrds"

def wrds_conn():
    user, pwd = wrds_creds()
    return psycopg2.connect(host=HOST, port=PORT, dbname=DB, user=user, password=pwd, sslmode="require")

if __name__ == "__main__":
    with wrds_conn() as conn, conn.cursor() as cur:
        cur.execute("select current_date;")
        print("WRDS ping OK:", cur.fetchone()[0])
