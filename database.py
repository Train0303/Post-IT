import sqlite3



def dbcon():
    return sqlite3.connect('test.db')

def create_table():
    try:
        db = dbcon()
        c = db.cursor()
        c.execute("CREATE TABLE user (userid varchar(20),password varchar(20))")
        db.commit()
    except Exception as e:
        print('db error:',e)
    finally:
        db.close()

def insert_data(userid,password):
    try:
        db = dbcon()
        c = db.cursor()
        setdata = (userid,password)
        c.execute("INSERT INTO user VALUES (?,?)",setdata)
        db.commit()
    except Exception as e:
        print('db error:',e)
    finally:
        db.close()

def login(userid,password):
    try:
        db = dbcon()
        c = db.cursor()
        result = c.execute("SELECT * from user where userid = '{0}' and password = '{1}'".format(userid,password))
        rows = c.fetchall()
        cnt = len(rows)
    except Exception as e:
        print('db error:',e)
    finally:
        db.close()
        return cnt

def gpu_check():
    try:
        db = dbcon()
        c = db.cursor()
        result = c.execute("SELECT status from gpu where now = 1")
        rows = c.fetchall()
        
    except Exception as e:
        print('db error:',e)
    finally:
        db.close()
        return rows[0][0]

def gpu_change(status):
    try:
        db = dbcon()
        c = db.cursor()
        c.execute("UPDATE gpu SET status = {0} where now = 1".format(status))
        db.commit()
    except Exception as e:
        print('db error:',e)
    finally:
        db.close()
