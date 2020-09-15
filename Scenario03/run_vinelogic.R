library(VineLOGIC)

i_dir<-file.path(getwd(),fsep = .Platform$file.sep)

jsons<-list.files(path = i_dir,pattern = '.json$')
for( i in jsons ) {
 l1 <-readLines(i)
 l2 <-gsub(pattern = '"Value": .', replace = '"Value": 0.', x = l1, fixed = TRUE)
 writeLines(l2, con=i)
}

output<-vines(start_date="2019-01-01",input_file_dir=i_dir,sim_length=900)

# state variable time series
states<-output[[1]]
df<-as.data.frame(states)

write.csv(df,file.path(getwd(),"vines_output.csv",fsep = .Platform$file.sep))

# summary variables
summary<-output[[2]]
df<-as.data.frame(summary)

write.csv(df,file.path(getwd(),"vines_summary.csv",fsep = .Platform$file.sep))
