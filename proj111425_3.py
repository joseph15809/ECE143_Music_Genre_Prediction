import csv
import h5py
import numpy as np
import hdf5_getters
import pandas as pd
import random
import string

bb_file="C:/Users/00inf/Downloads/ECE 143/proj143/ece143billboardhot1001958to2010.csv"
genre_file= "C:/Users/00inf/Downloads/ECE 143/proj143/msd-MAGD-genreAssignment.cls.txt"
mspath="C:/Users/00inf/Downloads/ECE 143/proj143/millionsongsubset"
msdsum="C:/Users/00inf/Downloads/ECE 143/proj143/msd_summary_file.h5"

#choose random song from the list
# i=random.randrange(1,273388)
bb_genre={}
bb_week=0
found=0 #complete find
msd_hit=0

wk=470
################## BILLBOARD FILE ########################
with open(bb_file,'r',newline=''):

    read1=pd.read_csv(bb_file,usecols=['song','artist'])

# while(bb_week<((len(read1))//100)):
    
    for song_num in range(100*wk+1,100*wk+23):
    
        bb_dict=dict(read1.loc[song_num])
        bb_song=bb_dict["song"]
        bb_artist=bb_dict["artist"]
        
        #print(bb_artist)
        #with ??
        #remove featuring, it is a plague
        substring="Featuring"
        substring2="&"
        ss3="(" #this may be problematic, well see
        ss4=","
        ss5="Pt"
        ss6="Part"
        ss7="Feat."
        ss8="["
        
        if substring in bb_artist:
            bb_artist=bb_artist.split(substring,1)[0]
            
        if substring2 in bb_artist:
            bb_artist=bb_artist.split(substring2,1)[0]
            
        if ss3 in bb_artist:
            bb_artist=bb_artist.split(ss3,1)[0]
            
        if ss4 in bb_artist:
            bb_artist=bb_artist.split(ss4,1)[0]
            
        if ss5 in bb_song:
            bb_song=bb_song.split(ss5,1)[0]
        
        if ss6 in bb_song:
            bb_song=bb_song.split(ss6,1)[0]
            
        if ss7 in bb_artist:
            bb_artist=bb_artist.split(ss7,1)[0]
            
        if ss8 in bb_artist:
            bb_artist=bb_artist.split(ss8,1)[0]
                

        
        ###
        print('\n', bb_song)
        print(bb_artist)
        
    
    
        #find song via artist name, song name, etc.
        h5 = hdf5_getters.open_h5_file_read(msdsum)
        ##### Using song found from billboard #########
        
        #do we need to worry about matching up to the encoding from dataset?(yes)
        translator = str.maketrans('', '', string.punctuation)

        # Remove punctuation
        bb_song_p = bb_song.translate(translator)
        
        
        enter_song="title==\""+bb_song_p+"\""
        idxs = h5.root.metadata.songs.get_where_list(enter_song)
        
        tr_id=[]
        flag_retry=0
        inner_flag=0
        
        if(len(idxs)==0):
            print('[None found of that song name!]')
            flag_retry=1
            
        else:
            for idx in idxs:
                art_name=h5.root.metadata.songs.cols.artist_name[idx]
                art_name_enc = bytes(str(art_name), "utf-8")
                
                if bb_artist.encode() in (art_name_enc): #could change this to ==
                    tr_id.append(h5.root.analysis.songs.cols.track_id[idx])

                else:
                    #print("nah")
                    inner_flag+=1
                    if(inner_flag==len(idxs)):
                        flag_retry=1
                    pass
        
        if(flag_retry==1):
            
            
            #do we need to worry about matching up to the encoding from dataset?(yes)
            translator = str.maketrans('', '', string.punctuation)

            # Remove punctuation
            bb_artist = bb_artist.translate(translator)
            
            if(bb_artist[-1]==' '):
                bb_artist2=bb_artist[:-1]
                bb_artist=bb_artist2
                print(bb_artist2)
            
            
            ## Do artist then song search instead?
            ##need to institute this when artist not found for a song name
            
            enter_artist="artist_name==\""+bb_artist+"\""
            idxs = h5.root.metadata.songs.get_where_list(enter_artist)
            print('[Pulling Artist Name]')
            
            if(len(idxs)!=0):
                for idx in idxs:
                    song_name=h5.root.metadata.songs.cols.title[idx]
                    # print(song_name)
                    song_name_enc = bytes(str(song_name), "utf-8")
                    
                    if bb_song.encode() in (song_name_enc): #could change this to ==
                        tr_id.append(h5.root.analysis.songs.cols.track_id[idx])
                        print('[SECOND TRY WORKED]')

                    else:
                        #print("nah")
                        pass
        h5.close()
        
        #im not sure why we are pulling multiple track id's in some cases
        #maybe print artist names to see whats up with that
        #print(tr_id)
        
        if(len(tr_id)!=0):
            msd_hit+=1
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
                print(genre_dict.get(tr_idn, "[Genre not found]"))
                genre_gotten=(genre_dict.get(tr_idn, "[Genre not found]"))
                

                
                if genre_gotten not in bb_genre:
                    if genre_gotten!='[Genre not found]':
                        bb_genre[genre_gotten]=1
                        found+=1 #found all fields for song
                else:
                    bb_genre[genre_gotten]=1+bb_genre[genre_gotten]
                    found+=1 #found all fields for song
                    
        
        else:
            print("[Artist not found for song name]")
        
        
    # bb_week+=1
    s_list=[]
    hold=[]
    s_list.append(f"Week {wk}")
    print(bb_genre)
    
    for key in (bb_genre):
        temp1=key
        temp2=bb_genre[key]/(found)
        hold.append(temp1)
        hold.append(temp2)
        s_list.append(hold.copy())
        hold.clear()
        
        
print(s_list)
print(msd_hit,'msd_hit')
print(found,'genre_found')