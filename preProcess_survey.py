def process_survey(data):
  """"Preprocess data based on encodings"""
  
  features = []
  
  # Encoding for age group
  age_group = data.get('age_group_5_years')
  if age_group == "18-29":  
      features.append(1)
  elif age_group == "30-34":
      features.append(2)
  elif age_group == "35-39":
      features.append(3)
  elif age_group == "40-44":
      features.append(4)
  elif age_group == "45-49":
      features.append(5)
  elif age_group == "50-54":
      features.append(6)
  elif age_group == "55-59":
      features.append(7)
  elif age_group == "60-64":
      features.append(8)
  elif age_group == "65-69":
      features.append(9)
  elif age_group == "70-74":
      features.append(10)
  elif age_group == "75-79":
      features.append(11)
  elif age_group == "80-84":
      features.append(12)
  else:
      features.append(13)

  # Encoding for race_eth
  race_value = data.get('race_eth')
  if race_value == "Non-Hispanic white":
      features.append(1)
  elif race_value == "Non-Hispanic black":
      features.append(2)
  elif race_value == "Asian/Pacific Islander":
      features.append(3)
  elif race_value == "Native American":
      features.append(4)
  elif race_value == "Hispanic":
      features.append(5)
  else:
      features.append(6)

  # Encoding for first degree relative
  first_degree = data.get('first_degree_hx')
  if first_degree == "No":
      first_degree = 0  
      features.append(0)
  else:
      first_degree = 1
      features.append(1)
      
  # Encoding for age menarche
  age_menarche = data.get('age_menarche')
  if age_menarche == "Age 14 or older":
      age_menarche = 0
      features.append(0)
  elif age_menarche == "Age 12-13":
      age_menarche = 1
      features.append(1)
  else:
      age_menarche = 2
      features.append(2)
      
  # Encoding for age first birth
  age_birth = data.get('age_first_birth')
  if age_birth == "Age < 20":
      age_birth = 0
      features.append(0)
  elif age_birth == "Age 20-24":
      age_birth = 1
      features.append(1)
  elif age_birth == "Age 25-29":
      age_birth = 2
      features.append(2)
  elif age_birth == "Age 30 or older":
      age_birth = 3
      features.append(3)
  else:
      age_birth = 4
      features.append(4)
      
  # Encoding for breast density
  birads_density = data.get('BIRADS_breast_density')
  if birads_density == "Almost entirely fat":
      birads_density = 1
      features.append(1)
  elif birads_density == "Scattered fibroglandular densities":
      birads_density = 2
      features.append(2)
  elif birads_density == "Heterogeneously dense":
      birads_density = 3
      features.append(3)
  else:
      birads_density = 4
      features.append(4)

  # Encoding for menopause
  menopause_status = data.get('menopaus')
  if menopause_status == "Pre-or peri-menopausal":
      menopause_status = 1
      features.append(1)
  else:
      menopause_status = 2
      features.append(2)
      
  # Encoding for bmi group
  bmi_group = data.get('bmi_group')
  if bmi_group == "10-24.99":
      bmi_group = 1
      features.append(1)
  elif bmi_group == "25-29.99":
      bmi_group = 2
      features.append(2)
  elif bmi_group == "30-34.99":
      bmi_group = 3
      features.append(3)
  else:
      bmi_group = 4
      features.append(4)
      
  # Encoding for biopsy
  biopsy = data.get('biophx')
  if biopsy == "No":
      features.append(0)
  else:
      features.append(1)

  # Encode derived features
  if (bmi_group >= 3) & (menopause_status >= 2):
    features.append(1)
  else:
      features.append(0)
  
  if (birads_density >= 3) & (first_degree == 1):
      features.append(1)
  else:
      features.append(0)

  if (age_menarche == 2) & (age_birth == 3):
      features.append(1)
  else:
      features.append(0)

  # Make sure all required features are present
  if len(features) != 12:
      return False
  else:
      return features