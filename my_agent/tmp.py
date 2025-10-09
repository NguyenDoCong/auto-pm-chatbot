from datetime import datetime
date = datetime.strptime('2014-12-04', '%Y-%m-%d').date()
print(type(date))