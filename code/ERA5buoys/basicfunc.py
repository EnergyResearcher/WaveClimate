import pandas
def round_to(n, roundto):
    return (round(n/roundto)*roundto)
    
def DM_to_DecDeg(DMcoordinate):
    #converting a string in a format DDMMMM to a float DD.DDDD
    degree = float(DMcoordinate[:-4])
    if degree > 0:
        minutes = float(f'{DMcoordinate[-4:-2]}.{DMcoordinate[-2:]}')
    else: 
        minutes = -float(f'{DMcoordinate[-4:-2]}.{DMcoordinate[-2:]}')
    #conversion based on
    # https://www.latlong.net/degrees-minutes-seconds-to-decimal-degrees
    return round_to(degree + minutes/60, 0.0001)



def proximity_calc(loc, source):
    #both args are tuples (lat, lon) in the same format
    #distance of loc from source
    return (loc[0]-source[0]).abs()/source[0], (loc[1]-source[1]).abs()/source[1]