import earthaccess
import sys
import argparse

def main():
    # pass the start and end dates in from the submitting script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date in YYYY-MM-DD format",
        required=True
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date in YYYY-MM-DD format",
        required=True
    )
    start_date = parser.parse_args().start_date
    end_date = parser.parse_args().end_date
    print('start date:', start_date)
    print('end date:', end_date)
    # do we need to login?
    login = True
    if login:
        print('logging in')
        auth = earthaccess.login()
    # do we want to see collections?
    see_collections = False
    if see_collections:
        print('getting collections')
        collections = earthaccess.search_datasets(
            short_name='MCD43A4'
        )
        # Step 2: See how many were returned
        print('Found {} collection(s)'.format(len(collections)))
        # Step 3: Inspect metadata for the first collection
        collection = collections[0]
        print('first collection:')
        print(collection)
        sys.exit()
        # Print summary information
        print("Title:", collection["umm"]["EntryTitle"])
        print("Version:", collection["umm"]["Version"])
        print("Start Date:", collection["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
        print("End Date:",collection["umm"]["TemporalExtent"]["RangeDateTime"]["EndingDateTime"])
    # download the data
    print('searching for data')
    bounding_box = (-124.73,25.85,-93.51,49.00)
    #start_date = '2003-01-01'
    #end_date = '2003-12-31'
    results = earthaccess.search_data(
        short_name='MCD43A4',
        version='061',
        temporal=(start_date,end_date),
        bounding_box=bounding_box
    )
    print('found {} data results'.format(len(results)))
    files = earthaccess.download(
        results,
        '/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_earthaccess'
    )
    results_quality = earthaccess.search_data(
        short_name='MCD43A2',
        version='061',
        temporal=(start_date,end_date),
        bounding_box=bounding_box
    )
    print('found {} quality results'.format(len(results_quality)))
    files = earthaccess.download(
        results_quality,
        '/scratch/users/trobinet/long_lfmc/trent_datasets/modis/modis_earthaccess'
    )

if __name__ == "__main__":
    main()
