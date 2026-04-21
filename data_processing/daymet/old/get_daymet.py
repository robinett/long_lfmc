import datetime
import subprocess

def main():
    start_date = datetime.date(2003,1,1)
    end_date = datetime.date(2004,1,1)
    current_date = start_date
    vars_to_get = ['tmax', 'tmin', 'prcp', 'srad', 'vp', 'dayl', 'swe']
    while current_date <= end_date:
        for var in vars_to_get:
            print('getting {} for {}'.format(var, current_date))
            subprocess.run(['bash',
                            'daymet.sh',
                            current_date.strftime('%Y'),
                            current_date.strftime('%m'),
                            current_date.strftime('%d'),
                            var],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        current_date += datetime.timedelta(days=1)


if __name__ == "__main__":
    main()
