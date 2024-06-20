import sqlite3

# connect to database
def get_db_connection():
    return sqlite3.connect("survey_database.db")

def commit_and_close(conn):
    conn.commit()
    conn.close()

# creating the database table
def create_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''CREATE TABLE IF NOT EXISTS survey_dataset_table
                   (age_group_5_years INTEGER NOT NULL,
                   race_eth INTEGER NOT NULL,
                   first_degree_hx INTEGER NOT NULL,
                   age_menarche INTEGER NOT NULL,
                   age_first_birth INTEGER NOT NULL,
                   BIRADS_breast_density INTEGER NOT NULL,
                   menopaus INTEGER NOT NULL,
                   bmi_group INTEGER NOT NULL,
                   biophx INTEGER NOT NULL,
                   high_bmi_post_menopause INTEGER NOT NULL,
                   family_hist_dense INTEGER NOT NULL,
                   menarche_first_birth INTEGER NOT NULL,
                   breast_cancer_history INTEGER NOT NULL
                       )''')
        print("Database table created successfully.")
    except sqlite3.Error as e:
        print("Error creating database table:", e)
    finally:
        commit_and_close(conn)

# adding the data from the survey to the database
def add_survey_data(dataDict):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''INSERT INTO survey_dataset_table
                          (age_group_5_years, race_eth, first_degree_hx, age_menarche, age_first_birth, 
                          BIRADS_breast_density, menopaus, bmi_group, 
                          biophx, high_bmi_post_menopause, family_hist_dense, 
                          menarche_first_birth, breast_cancer_history ) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (dataDict["age_group_5_years"], dataDict["race_eth"], dataDict["first_degree_hx"], dataDict["age_menarche"], dataDict["age_first_birth"],
                        dataDict["BIRADS_breast_density"], dataDict["menopaus"], dataDict["bmi_group"],
                        dataDict["biophx"], dataDict["high_bmi_post_menopause"], dataDict["family_hist_dense"],
                        dataDict["menarche_first_birth"], dataDict["breast_cancer_history"]))
        print("Data added successfully.")
    except sqlite3.Error as e:
        print("Error adding data:", e)
    finally:
        commit_and_close(conn)