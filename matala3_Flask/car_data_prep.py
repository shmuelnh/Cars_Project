import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



def prepare_data(df):
   #הורדנו עמודות שלא שימושיות לחיזוי
    df = df.drop(['Cre_date','Repub_date','Test','Supply_score'], axis=1)


    # ... (rest of the function)
    ### manufactor
    ### נמנע כפילויות
    df['manufactor'] = df['manufactor'].str.replace('Lexsus', 'לקסוס')
    
    ### Km
    ### נכפיל ב1000 ערכים תלת ספרתיים
    ### כשמשתמש כותב למשל 100 הוא מתכוון ל100 אלף
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    def multiply_if_three_digits(km):
        if pd.notnull(km) and len(str(int(km))) == 3:
            return km * 1000
        return km
    df['Km'] = df['Km'].apply(multiply_if_three_digits)
    
    ### יצרנו מילון של כל שנה וממוצע הקילומטר שלה על ידי group by מנתוני האימון
    ### נמלא כל ערך חסר בערך מהמילון
    km_mean_by_year = {1983.0: 100000.0, 1988.0: 200000.0, 1990.0: 305000.0, 1995.0: 140000.0, 1998.0: 250000.0, 1999.0: 173333.33333333334, 2000.0: 417000.0, 2002.0: 299000.0, 2003.0: 400000.0, 2004.0: 155000.0, 2005.0: 239500.0, 2006.0: 165890.0, 2007.0: 245380.95238095237, 2008.0: 181603.6976744186, 2009.0: 167795.60975609755, 2010.0: 187437.3492063492, 2011.0: 161023.5593220339, 2012.0: 171566.2837837838, 2013.0: 170212.19444444444, 2014.0: 147844.8231292517, 2015.0: 132498.14814814815, 2016.0: 119682.68702290076, 2017.0: 104210.79411764706, 2018.0: 87105.94852941176, 2019.0: 54445.49367088608, 2020.0: 58640.54054054054, 2021.0: 19237.5, 2022.0: 12400.0, 2023.0: 6351.0}
    df['Km'] = df.apply(lambda row: km_mean_by_year[row['Year']] if pd.isnull(row['Km']) else row['Km'], axis=1)
    
    ### Prev_ownership
    ### אם זה יד ראשונה אז אין בעלים קודמים
    df.loc[df['Hand'] == 1, 'Prev_ownership'] = 'אין'
    df['Prev_ownership'] = df['Prev_ownership'].fillna('לא מוגדר')
    
    ### Curr_ownership
    ### "אם כתוב בפירוט "פרטי/ת" אז בבעלות הנוכחית נמלא "פרטית
    ### וכן הלאה לכל סוג בעלות
    df.loc[df['Description'].str.contains('פרטי|פרטית', case=False), 'Curr_ownership'] = 'פרטית'
    df.loc[df['Description'].str.contains('השכרה', case=False),'Curr_ownership'] = 'השכרה'
    df.loc[df['Description'].str.contains('חברה', case=False),'Curr_ownership'] = 'חברה'
    df.loc[df['Description'].str.contains('ליסינג', case=False),'Curr_ownership'] = 'ליסינג'
    df.loc[df['Description'].str.contains('מונית', case=False),'Curr_ownership'] = 'מונית'
    ###את השאר המלא כלא מוגדר
    df['Curr_ownership'] = df['Curr_ownership'].fillna('לא מוגדר')


    ### model
    ### נקבץ ערכים זהים כדי למנוע כפילויות
    df['model'] = df['model'].str.replace('CIVIC','סיוויק')
    df['model'] = df['model'].str.replace('ACCORD','אקורד')
    df['model'] = df['model'].str.replace('C-Class קופה','C-CLASS קופה')
    df['model'] = df['model'].str.replace('E- CLASS','E-Class')
    df['model'] = df['model'].str.replace('JAZZ','ג\'אז')
    df['model'] = df['model'].str.replace('ג\'אז','ג`אז')
    ### הורדת השנה מעמודת מודל
    df['model'] = df['model'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    ### הורדת שם יצרן מעמודת מודל
    def clean_model_name(row):
        manufactor = row['manufactor']
        model = row['model']
        cleaned_model = model.replace(manufactor, "").strip()
        return cleaned_model
    df['model'] = df.apply(clean_model_name, axis=1)
    
    ###Gear
    ###קיבוץ אוטומטית לאוטומט
    df['Gear'] = df['Gear'].replace({'אוטומטית': 'אוטומט'})
    #עמודת גיר מכילה ערך חסר אחד. נמלא אותו
    df['Gear'] = df['Gear'].fillna('לא מוגדר')
    
    ###capacity_Engine
    ### הורדת פסיקים והמרה למספר
    df['capacity_Engine'] = df['capacity_Engine'].str.replace(',', '')
    df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')

    ###capacity_Engine מה שנשאר מ 
    ### נשלים בערך הממוצע
    most_frequent_imputer = SimpleImputer(strategy='mean')
    df[['capacity_Engine']] = most_frequent_imputer.fit_transform(df[['capacity_Engine']])
    
    ###Engine_type
    ### קיבוץ היבריד להיברידי
    df['Engine_type'] = df['Engine_type'].replace({'היבריד': 'היברידי'})
    # מה שנשאר נשלים בערך השכיח
    most_frequent_imputer = SimpleImputer(strategy='most_frequent')
    df[['Engine_type']] = most_frequent_imputer.fit_transform(df[['Engine_type']])

    
    ###Pic_num
    ### אם יש ערך חסר זאת אומרת שאין תמונות - נמלא ב0
    df['Pic_num'] = df['Pic_num'].fillna(0) 
   

    ###City
    ### נתקן ערים שגויות על פי ההגיון
    df['City'] = df['City'].str.strip()
    corrections = {
        'רמת': 'רמת גן',
        'באקה': 'באקה אל-גרביה',
        'באקה א שרקיה': 'באקה אל-גרביה',
        'פרדס': 'פרדס חנה-כרכור',
        'חיפ': 'חיפה',
        'תל': 'תל אביב',
        'קרית': 'קרית אתא',
        'כפר': 'כפר סבא',
        'הוד': 'הוד השרון',
        'פתח תקווה,יהוד': 'פתח תקווה',
        'תל אבייב': 'תל אביב',
        'פ"ת': 'פתח תקווה',
        'בת': 'בת ים',
        'גבעתיי': 'גבעתיים',
        'ראשון': 'ראשון לציון',
        'ראש': 'ראש העין',
        'חד': 'חד-נס',
        'רא': '',  # Removing as it's unclear what it refers to
        'מזכרת': 'מזכרת בתיה',
        'נתנייה': 'נתניה',
        'jeruslem': 'ירושלים',
        'Rehovot': 'רחובות',
        'haifa': 'חיפה',
        'Tel aviv': 'תל אביב',
        'ashdod': 'אשדוד',
        'Rishon LeTsiyon': 'ראשון לציון',
        'Tzur Natan': 'צור נתן',
        'ק.אתא': 'קרית אתא',
        'נצרת עילית': 'נוף הגליל',
        'קריית': 'קרית',
        'פתח תיקווה': 'פתח תקווה',
        'נהרייה': 'נהריה',
        'מעלות': 'מעלות-תרשיחא'
        }
    df['City'] = df['City'].replace(corrections)
    
    #שינוי של עמודת "אזור" לאזורים יותר גדולים
    # חלוקה לאזורים בארץ
    def create_city_to_area_mapping():
        city_to_area = {
        'צפון': ['כרמיאל', 'נהריה', 'עכו', 'צפת', 'קרית שמונה', 'טבריה', 'נצרת', 'עפולה', 'מגדל העמק', 'בית שאן', 'נוף הגליל', 'ראש פינה', 'מגדל', 'קצרין', 'מעלות-תרשיחא', "בית ג'ן", 'מגאר', 'עין מאהל', 'כפר תבור', 'ריחאניה', 'יובלים', 'גילון', 'אילון', 'חד נס', 'ארבל', 'כמון', 'פקיעין', 'מצפה נטופה ד.נ. גליל תחתון', 'סאגור', 'נאעורה', 'סלמה', 'חיפה', 'טירת כרמל', 'קרית אתא', 'קרית מוצקין', 'קריית ים', 'נשר', 'עתלית', 'קרית ביאליק', 'קריית אתא','טירת הכרמל','גבעת אבני', 'קריית ביאליק', 'קריות', 'החותרים', 'קרית ים', 'ירכא', "סח'נין", 'ריינה', 'עראבה', 'אעבלין', 'מרר', 'שפרעם', 'כאבול', 'כפר מנדא', 'דאלית אל כרמל', 'אבו סנאן', 'נחף', "מג'ד אל-כרום", 'כפר כנא', 'דבוריה', 'כסרא', 'טמרה', 'מזרעה', 'יוקנעם', 'יקנעם עילית', 'שריד', 'תמרת', 'גבעת אלה', 'רמת ישי', 'קרית טבעון', 'קריית טבעון', 'עספיא', 'חד-נס', 'אלמגור', 'בוקעתא', 'אבני איתן', 'נווה אור', 'עכו', 'עראבה', 'חריש', 'רכסים', "בית ג'אן", 'פוריה','זרזיר','גשר הזיו','מולדת','רמת מגשימים','יוקנעם עילית','מושב מולדת', 'מעלות תרשיחא'],

        'דרום': ['כפר הרי"ף','נחלה','יד בנימין','באר שבע', 'אילת', 'דימונה', 'שדרות', 'אופקים', 'נתיבות', 'קרית גת', 'עומר', 'להבים', 'חורה', 'שגב שלום', 'תל שבע', 'מיתר', 'רהט', 'כסיפה', 'מעגלים', 'עוצם', 'אשקלון', 'קרית מלאכי', 'אשדוד', 'ברוש', 'שריגים', 'ערד', 'ירוחם', 'מצפה רמון', 'אילת השחר'],

        'מרכז': ['מגשימים','בית עוזיאל','רמת גן', 'רחובות', 'ראשון לציון', 'פתח תקווה', 'בת ים', 'חולון', 'גדרה', 'נס ציונה', 'בני ברק','תנובות', 'תל אביב יפו', 'גבעת שמואל', 'גבעתיים', 'אור יהודה', 'מודיעין מכבים רעות', 'מודיעין', 'אלעד', 'קרית אונו', 'יהוד מונוסון', 'אזור', 'יבנה', 'גני תקווה', 'באר יעקב', 'מזכרת בתיה', 'קרית עקרון', 'רמת השרון', 'בית דגן', 'ברקת', 'שוהם', 'אליכין', 'גן יבנה', 'תל אביב', 'ראש העין', 'לוד', 'רמלה', 'קדרון', 'זכריה', 'נחושה', 'שתולים', 'כפר חב"ד', 'צפריה', 'ניר צבי', 'נגבה', 'חדרה', 'אור עקיבא', 'גבעתי', 'יהוד', 'בית', 'פתח', 'בארותיים', 'צפריה', 'אבני איתן', 'כפר מצר', 'בית קשת', 'כפר מנחם', 'ניר צבי', 'כפר עגר', 'בניה', 'אומן'],

        'השרון': ['רעננה', 'אבן יהודה', 'כפר סבא', 'נתניה', 'פרדס חנה-כרכור', 'תל מונד', 'הוד השרון', 'הרצליה', 'כפר יונה', 'קיסריה', 'זכרון יעקב', 'בנימינה גבעת עדה', 'פוריידיס', 'פרדס חנה כרכור', 'כפר יעבץ', 'עזריאל', 'משמר השרון', 'שער אפרים', 'אחיטוב', 'חופית', 'גן השומרון', 'רשפון', 'צור יצחק', 'כוכב יאיר', 'מתן', 'קדימה צורן', 'צור נתן', 'תל יצחק', 'מכמורת', 'חרוצים', 'עמק חפר', 'גבעת חיים מאוחד', 'אודים', 'טייבה משולש', 'אחיעזר', 'אבן', 'גבעת כ"ח', 'טייבה', 'גבעת עדה', 'טירה', 'קלנסווה', 'כפר קאסם', 'באקה אל-גרביה', 'זמר', 'ערערה', 'כפר קרע', 'אום אל פחם'],

        'ירושלים והסביבה': ['ירושלים', 'בית שמש', 'מעלה אדומים', 'קרית יערים', 'מבשרת ציון', 'צור הדסה', 'ביתר עילית', 'אורה', 'עטרת', 'גבע בנימין', 'בית זית', 'אבו גוש', 'פסגת זאב', 'עזריה', 'מודיעין עילית'],

        'יהודה ושומרון': ['אורנית', 'עץ אפרים', 'עלי זהב', 'חשמונאים', 'רבבה', 'שערי תקווה', 'אלעזר', 'קרית ארבע', 'גבעת זאב', 'כפר תפוח', 'סלעית', 'מתתיהו', 'בת עין', 'אריאל', 'קרני שומרון', 'אלקנה', 'עלי זהב', 'חיננית', 'קציר']
        }
    
    # יצירת מילון הפוך: מעיר לאזור
        return {city: area for area, cities in city_to_area.items() for city in cities}

    # יצירת המיפוי
    city_to_area_mapping = create_city_to_area_mapping()

    # הפונקציה שתמפה ערים לאזורים
    def map_city_to_area(city):
        return city_to_area_mapping.get(city, 'לא ידוע')

    # הפעלת הפונקציה על עמודת ה-city
    df['Area'] = df['City'].apply(map_city_to_area)
    
    ###Color
    ### השלמת הצבע מתוך התיאור
    #צבעים יחודיים מעמודת הצבעים: 
    colors = ['כחול כהה מטאלי', 'כחול בהיר', 'אפור מטאלי', 'שחור', 'חום',
    'כסוף', 'לבן', 'לבן מטאלי', 'לבן פנינה', 'אפור עכבר', 'אפור',
     'כחול', 'סגול', 'אדום', 'כסף מטלי', 'כתום', 'לבן שנהב',
    'סגול חציל', 'כסוף מטאלי', 'כחול בהיר מטאלי', 'טורקיז', 'כחול כהה',
    "בז'", 'בורדו', 'ירוק', 'שמפניה', 'ירוק מטאלי', 'תכלת',
    'חום מטאלי', 'אדום מטאלי', 'כחול מטאלי', "בז' מטאלי", 'ורוד',
    'ברונזה', 'ירוק בהיר', 'זהב מטאלי', 'תכלת מטאלי', 'זהב']


    def fill_color(row):
        if pd.isna(row['Description']):
            return row['Color']
        for color in colors:
            if color in row['Description']:
                return color
        return row['Color']

    # יישום הפונקציה על ה-DataFrame
    df['Color'] = df.apply(fill_color, axis=1)
    
    # נרצה לשמור על יחס הצבעים של הרכבים בשביל מודל החיזוי
    #יצרנו מילון עם כל צבע והיחס שלו מתוך כל הצבעים
    #  יצרנו את המילון הזה מתוך פונקציה שהרצנו על נתוני האימון
    # (מצורפת הפונקציה)
    ##Create a dictionary to map 'Year' to mean 'Km'
    #km_mean_by_year = df.groupby('Year')['Km'].mean()
    #km_mean_mapping = km_mean_by_year.to_dict()

    #Create a dictionary to map 'Year' to mean 'Km'
    km_mean_by_year = df.groupby('Year')['Km'].mean()
    color_dict = {'שחור': 0.23127463863337713, 'לבן': 0.22207621550591328, 'אפור מטאלי': 0.09198423127463863, 'אפור': 0.07227332457293036, 'כסוף': 0.05781865965834428, 'לבן פנינה': 0.04467805519053877, 'אפור עכבר': 0.03942181340341656, 'כחול': 0.03153745072273324, 'כחול כהה מטאלי': 0.030223390275952694, 'כסוף מטאלי': 0.024967148488830485, 'לבן שנהב': 0.023653088042049936, 'לבן מטאלי': 0.01971090670170828, 'אדום': 0.010512483574244415, 'כסף מטלי': 0.010512483574244415, 'ירוק': 0.00788436268068331, 'זהב מטאלי': 0.00788436268068331, 'חום': 0.00788436268068331, 'כחול כהה': 0.006570302233902759, 'תכלת': 0.005256241787122208, "בז' מטאלי": 0.005256241787122208, 'תכלת מטאלי': 0.003942181340341655, 'ירוק בהיר': 0.003942181340341655, 'שמפניה': 0.003942181340341655, 'סגול חציל': 0.003942181340341655, 'כחול בהיר': 0.003942181340341655, 'חום מטאלי': 0.003942181340341655, 'זהב': 0.003942181340341655, 'טורקיז': 0.002628120893561104, 'כחול בהיר מטאלי': 0.002628120893561104, 'בורדו': 0.002628120893561104, 'אדום מטאלי': 0.002628120893561104, 'ברונזה': 0.002628120893561104, 'כחול מטאלי': 0.002628120893561104, 'כתום': 0.001314060446780552, 'ורוד': 0.001314060446780552, "בז'": 0.001314060446780552, 'ירוק מטאלי': 0.001314060446780552}


    
    def fill_missing_colors(df, color_dict):
        # המרת המילון לסדרה של pandas
        color_counts = pd.Series(color_dict)
    
        # נרמול הערכים כדי להבטיח שהם מסתכמים ל-1
        color_counts = color_counts / color_counts.sum()
    
        # מציאת השורות החסרות
        missing_color_indices = df[df['Color'].isna()].index
    
        # מילוי השורות החסרות לפי היחסים מ-color_dict
        for idx in missing_color_indices:
            df.at[idx, 'Color'] = np.random.choice(color_counts.index, p=color_counts.values)
    
        return df

    # יישום הפונקציה על ה-DataFrame
    df = fill_missing_colors(df, color_dict)
    
    # Convert 'Year' to numeric, replacing any non-numeric values with NaN
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    #df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    #יצירת משתנים חדשים
    df['Age'] = 2024 - df['Year']  # גיל הרכב
    df['Km_per_year'] = df['Km'] / df['Age'] # קילומטראז' שנתי ממוצע
    
    
    #convert to float
    df['Year'] = df['Year'].astype(float)
    df['Hand'] = df['Hand'].astype(float)
    df['Age'] = df['Age'].astype(float)
    
    #חלוקה לפיצ'רים קטגוריאלים ומספריים
    categorical_columns = [
    'manufactor', 'model', 'Gear', 'Engine_type',
    'Prev_ownership', 'Curr_ownership', 'Area','City',
    'Color']

    numerical_columns = [
    'Year', 'Hand', 'capacity_Engine',
    'Pic_num', 'Km','Km_per_year']

    
    ### הורדנו עמודות לא רלוונטיות או עם הרבה ערכים חסרים:
    #### 'Cre_date','Repub_date','Description' - לא רלוונטי לחיזוי
    #### 'Test','Supply_score' - יותר מידי ערכים חסרים - לא ניתן להסיק מכמות נתונים זו
    #### 'Age' - קורלציה גבוהה עם year
    #### We used Age just for calculate the Km_per_year column
    df = df.drop(['Description','Age'], axis=1)
    ###סטנדרטיזציה
    ###יצירת סקאלה סטנדרטית לערכים המספריים
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    ### one hot encoder
    #### למשתנים הקטגוריאלים
    
    def create_one_hot_encoder(df, categorical_columns):

        # Initialize the OneHotEncoder
        onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        #drop = 'first'
        # Create the ColumnTransformer
        ct = ColumnTransformer(
        [('onehot', onehot, categorical_columns)],
        remainder='passthrough'
        )

        # Fit and transform the data
        encoded_array = ct.fit_transform(df)

        # Get the new column names
        onehot_columns = ct.named_transformers_['onehot'].get_feature_names_out(categorical_columns)
        other_columns = df.columns.drop(categorical_columns).tolist()
        new_columns = onehot_columns.tolist() + other_columns

        # Create a new dataframe with encoded data
        encoded_df = pd.DataFrame(encoded_array, columns=new_columns, index=df.index)

        return ct, encoded_df

    categorical_columns =  [
    'manufactor', 'model', 'Gear', 'Engine_type',
    'Prev_ownership', 'Curr_ownership', 'Area',
    'Color','City']

    ct, encoded_df = create_one_hot_encoder(df, categorical_columns)
    df = encoded_df


    X = df.drop(['Price'],axis=1)
    y = df['Price']
    
    ###Feature Selection:
    
    def backward_elimination(X, y, significance_level=0.05):
        features = list(X.columns)
    
        while len(features) > 0:
            # חישוב F-statistics ו-p-values
            f_stats, p_values = f_regression(X[features], y)
        
            # מציאת ה-p-value הגבוה ביותר
            max_p_value = p_values.max()
        
            if max_p_value > significance_level:
                # מציאת המאפיין עם ה-p-value הגבוה ביותר
                excluded_feature = features[p_values.argmax()]
            
                # הסרת המאפיין מהרשימה
                features.remove(excluded_feature)
            else:
                # אם כל ה-p-values מתחת ל-significance level, עוצרים
                break
    
        return features
    # שימוש בפונקציה
    selected_features = backward_elimination(X, y)
    
    ###הגדרת x ו y
    X = df[selected_features]
    y = df['Price']
    ### איחוד לדאטה פריים אחד
    def combine_features_and_target(X, y):
    # Create a copy of X to avoid modifying the original DataFrame
        combined_df = X.copy()
    
        # Add the target variable (y) to the DataFrame
        combined_df['Price'] = y
    
        return combined_df

    # Usage
    result_df = combine_features_and_target(X, y)
    
    return result_df