library(NLP)
library(tm)
require(graphics)
data <- Corpus(DirSource("./recipes"))

limpio = tm_map(data, removeWords, stopwords("english"))
limpio = tm_map(limpio, content_transformer(tolower))
limpio = tm_map(limpio, removeNumbers)
limpio = tm_map(limpio, removePunctuation)
limpio = tm_map(limpio, stripWhitespace)

limpio = tm_map(limpio, stemDocument)

dtm = DocumentTermMatrix(limpio)
#dtm = weightTfIdf(dtm)
m = as.matrix(dtm)

distancias = dist(dtm)
agrupamiento = hclust(distancias)
clusterw = hclust(distancias, method = "ward.D")
#plot(agrupamiento, -1)
plot(agrupamiento, hang = -1)
plot(clusterw, hang = -1)
#grupos = cutree(agrupamiento, k = 5)

#plot(agrupamiento, hang = 0.1, cex = 0.7, main = "documentos")
#rect.hclust(agrupamiento, k = 5, border = "red")


#kmeans example


#kfit = kmeans(distancias, 2, nstart = 200)
#library(cluster)
#clusplot(as.matrix(distancias), kfit$cluster, color = T, shade = T, labels = 2, lines = 0)

#wss <- 2:29

#for( i in 2:29) wss[i] <- sum(kmeans(distancias, centers = i, nstart = 25)$withinss)
#plot(2:29, wss[2:29], type = "b", xlab = "Number of clusters", ylab = "within groups sum of squares")

