import csv
import h5py
import numpy as np
import hdf5_getters
import pandas as pd
import random

bb_file="C:/Users/00inf/Downloads/ECE 143/proj143/ece143billboardhot1001958to2010.csv"
genre_file= "C:/Users/00inf/Downloads/ECE 143/proj143/msd-MAGD-genreAssignment.cls.txt"
mspath="C:/Users/00inf/Downloads/ECE 143/proj143/millionsongsubset"
msdsum="C:/Users/00inf/Downloads/ECE 143/proj143/msd_summary_file.h5"

#choose random song from the list
i=random.randrange(1,273388)

#20010; 20027; 20139; 20216; 23607; 24666; 24601

#20005; 23457
#Dataset contains 1922 - 2011

#2000s-4000s ~ 1/10 success
#15000s - better - genre not found often
#20000s - 3 years back
#35k - 7 years back
#45k -  9 years back, getting more consistent
################## BILLBOARD FILE ########################
with open(bb_file,'r',newline=''):

    read1=pd.read_csv(bb_file,usecols=['song','artist'])
    bb_dict=dict(read1.loc[i])
    #print(bb_dict["song"])
    bb_song=bb_dict["song"]
    bb_artist=bb_dict["artist"]
    
    #print(bb_artist)
    #with ??
    #remove featuring, it is a plague
    substring="Featuring"
    substring2="&"
    ss3="(" #this may be problematic, well see
    
    if substring in bb_artist:
        bb_artist=bb_artist.split(substring,1)[0]
        
    if substring2 in bb_artist:
        bb_artist=bb_artist.split(substring2,1)[0]
        
    # if ss3 in bb_song:
    #     bb_song=bb_song.split(ss3,1)[0]
        
    #do we need to worry about matching up to the encoding from dataset?(yes)
    print(bb_song)
    print(bb_artist)
    


#find song via artist name, song name, etc.
h5 = hdf5_getters.open_h5_file_read(msdsum)
##### Using song found from billboard #########

enter_song="title==\""+bb_song+"\""
idxs = h5.root.metadata.songs.get_where_list(enter_song)

tr_id=[]

if(len(idxs)==0):
    print('None found of that song name!')
else:
    for idx in idxs:
        art_name=h5.root.metadata.songs.cols.artist_name[idx]
        art_name_enc = bytes(str(art_name), "utf-8")
        
        if bb_artist.encode() in (art_name_enc): #could change this to ==
            tr_id.append(h5.root.analysis.songs.cols.track_id[idx])
        else:
            #print("nah")
            pass
h5.close()

#im not sure why we are pulling multiple track id's in some cases
#maybe print artist names to see whats up with that
#print(tr_id)

if(len(tr_id)!=0):
    tr_idn=tr_id[0].decode("utf-8")
    print(tr_idn)

        
    ##################### genre file ###################################
    # we would use this after weve found the desired songfile in the database
    
    with open(genre_file,'r',newline='') as genfile:
        reader=csv.reader(genfile, delimiter='\t')
        genre_list=[]
        for row in reader:
            genre_list.append(row)
        genre_dict=dict(genre_list)
        #Looks good, we have a dict with some song IDs and the genre
        print(genre_dict.get(tr_idn, "Genre not found"))

else:
    print("Artist not found for song name")
