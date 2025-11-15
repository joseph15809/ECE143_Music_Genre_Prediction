import csv
import h5py
import numpy as np
import hdf5_getters
import pandas as pd
import random
import string


from datetime import datetime

# Get the current date and time
now = datetime.now()

# Format the time as a string (e.g., "HH:MM:SS")
current_time = now.strftime("%H:%M:%S")

# Print the formatted time
print("Current Time = \n \n ", current_time)



bb_file="C:/Users/00inf/Downloads/ECE 143/proj143/ece143billboardhot1001958to2010.csv"
genre_file= "C:/Users/00inf/Downloads/ECE 143/proj143/msd-MAGD-genreAssignment.cls.txt"
mspath="C:/Users/00inf/Downloads/ECE 143/proj143/millionsongsubset"
msdsum="C:/Users/00inf/Downloads/ECE 143/proj143/msd_summary_file.h5"
genre_2="C:/Users/00inf/Downloads/ECE 143/proj143/msd_tagtraum_cd2.cls"

master_list=[]

for wk in range(400,480,10):

    #choose random song from the list
    # i=random.randrange(1,273388)
    bb_genre={}
    #bb_week=0
    found=0 #complete find
    msd_hit=0
    
    ################## BILLBOARD FILE ########################
    with open(bb_file,'r',newline=''):
    
        read1=pd.read_csv(bb_file,usecols=['song','artist'])
    
    # while(bb_week<((len(read1))//100)):
        
        for song_num in range(100*wk+0,100*wk+10):
        
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
            print('\n')
            print(bb_song)
            print(bb_artist)
            
        
            #find song via artist name, song name, etc.
            h5 = hdf5_getters.open_h5_file_read(msdsum)
            ##### Using song found from billboard #########
            
            #do we need to worry about matching up to the encoding from dataset?(yes)
            translator = str.maketrans('', '', string.punctuation)
    
            # Remove punctuation
            bb_song_p = bb_song.translate(translator)
            
            
            enter_song="title==\""+bb_song_p+"\""
            
            #         SONG TITLE SEARCH
            # requires an exact match for case. Does not accept certain punctuation
            # ** optimize for punctuation that is and isnt accepted
            idxs = h5.root.metadata.songs.get_where_list(enter_song)
            
            tr_id=[] #track id number
            flag_retry=0 #to manage not finding the track
            inner_flag=0 #to manage not finding the track
            
            if(len(idxs)==0): #no track list returned
                print('[None found of that song name!]')
                flag_retry=1
                
            else: #track list matching title was returned
                for idx in idxs: #look for desired artist amongst track list
                    art_name=h5.root.metadata.songs.cols.artist_name[idx]
                    art_name_enc = bytes(str(art_name), "utf-8")
                    
                    if bb_artist.encode() in (art_name_enc): #could change this to == (may help)
                        tr_id.append(h5.root.analysis.songs.cols.track_id[idx]) #if song found, add to list of IDs that match this track
    
                    else: #track does not match desired track
                        inner_flag+=1
                        if(inner_flag==len(idxs)): #if entire track list does not have desired track
                            flag_retry=1 #try to look via artist instead
                        pass
            
            if(flag_retry==1): #if song title search was unsuccessful
                # Remove punctuation
                translator = str.maketrans('', '', string.punctuation)
                bb_artist = bb_artist.translate(translator)
                
                #Make sure the artist name doesnt have a space at the end
                if(bb_artist[-1]==' '):
                    bb_artist2=bb_artist[:-1]
                    bb_artist=bb_artist2
                    #print(bb_artist2)
                
                ## Do artist then song search instead
                enter_artist="artist_name==\""+bb_artist+"\""
                idxs = h5.root.metadata.songs.get_where_list(enter_artist)
                print('[Pulling Artist Name...]')
                
                if(len(idxs)!=0):
                    for idx in idxs:
                        song_name=h5.root.metadata.songs.cols.title[idx]
                        # print(song_name)
                        song_name_enc = bytes(str(song_name), "utf-8")
                        
                        if bb_song.encode() in (song_name_enc): #could change this to ==
                            tr_id.append(h5.root.analysis.songs.cols.track_id[idx])
                            #print('[SECOND TRY WORKED]')
                        else:
                            pass
            h5.close()
            
            #im not sure why we are pulling multiple track id's in some cases (multiple recordings / listings of same song)
            
            if(len(tr_id)!=0): #if track id was found
                msd_hit+=1 #hit from million songs dataset
                tr_idn=tr_id[0].decode("utf-8") #use any one of the track ids
                print(tr_idn) 
            
                ##################### genre file ###################################
                # After weve found the desired songfile in the database we correlate to genre
                # via genre-trackid files
                
                with open(genre_file,'r',newline='') as genfile:
                    reader=csv.reader(genfile, delimiter='\t')
                    genre_list=[]
                    for row in reader: #iterating through file
                        if(tr_idn in row):
                            genre_list.append(row)
                    genre_dict=dict(genre_list)
                    
                    #Looks good, we have a dict with some song IDs and the genre
                    print(genre_dict.get(tr_idn, "[Genre not found]"))
                    genre_gotten=(genre_dict.get(tr_idn, "[Genre not found]"))
                    
                    #if genre not already in genre counting list
                    if genre_gotten not in bb_genre:
                        if genre_gotten!='[Genre not found]':
                            bb_genre[genre_gotten]=1
                            found+=1 #found all fields for song
                            
                        else:
                            ############# genre file #2 #################
                            #same thing, searching for track id in file and grabbing genres
                            final_dict={}
                            
                            with open(genre_2,'r',newline='') as genfile2:
                                            reader2=csv.reader(genfile2, delimiter='\t')
                                            #a bunch of likely unneeded lists
                                            genre_list2=[]
                                            gl2_slist=[]
                                            gl2_slist2=[]
                                            gen_list3=[]
                                            gen_wlist=[]
                                            final_list=[]
                                            
                                            for ind2,row in enumerate(reader2):
                                
                                                if(tr_idn in row):
                                                
                                                    for index,item in enumerate(row): #due to formatting of data
                                                    
                                                        gl2_slist=item.split('\t')
                                                        
                                                        if(index==0):
                                                            gl2_slist2.append(gl2_slist[0])
                                                        else:
                                                            gen_list3.append(item)
                                                            
                                                    gen_wlist.append(gen_list3.copy())
                                                    gen_list3.clear()
                                                    final_list=list(zip(gl2_slist2,gen_wlist))
                                                    final_dict=dict(final_list)
                                                    #print(final_dict,'final_dict')
                            
                            #either genre is found or not found...
                            print(final_dict.get(tr_idn, "[Genre not found AGAIN]"))
                            genre_gotten=(final_dict.get(tr_idn, "[Genre not found]"))
                            
                            for i in genre_gotten: #have to iterate cause multiple genres 
                                                    #from this file (stored in list)
                                if (i not in bb_genre) and (genre_gotten!='[Genre not found]'):                                
                                    bb_genre[i]=(1/len(genre_gotten)) #weighting the influence of one song on the numbers
                                    
                                elif(i in bb_genre) and (genre_gotten!='[Genre not found]'):
                                    bb_genre[i]=(1/len(genre_gotten))+bb_genre[i] #weighting the influence of one song on the numbers
                                    
                            if(genre_gotten!='[Genre not found]'):
                                found+=1 #found all fields for song
                    else:
                        bb_genre[genre_gotten]=1+bb_genre[genre_gotten]
                        found+=1 #found all fields for song
                        
            else:
                print("[Artist not found for song name]")
            
            
        # bb_week+=1
        
        #writing final results to weekly summary
        s_list=[]
        hold=[]
        s_list.append(f"Week {wk}")
        #print(bb_genre)
        
        for key in (bb_genre):
            temp1=key
            temp2=bb_genre[key]/(sum(bb_genre.values()))
            hold.append(temp1)
            hold.append(temp2)
            s_list.append(hold.copy())
            hold.clear()
            
    #write results to final summary list
    print(s_list)
    print(msd_hit,'msd_hit')
    print(found,'genre_found')
    master_list.append(s_list.copy())
    
    s_list.clear()

print(master_list)
# Get the current date and time
now = datetime.now()

# Format the time as a string (e.g., "HH:MM:SS")
current_time = now.strftime("%H:%M:%S")

# Print the formatted time
print("\n \n Current Time =", current_time)