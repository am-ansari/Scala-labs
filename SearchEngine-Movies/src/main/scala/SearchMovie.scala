import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}

object SearchMovie {
  def main(args: Array[String]): Unit = {
    var logger :StringBuilder = new StringBuilder()

    if (args.length != 5) {
      println("Usage: SearchMovie <plotSummary> <movieMetadata>" +
        " <searchSingleTerms> <searchMultipleTerms> <logger> <output1> <output2>")
    }
    // create Spark context with Spark configuration
    val sc = new SparkContext(new SparkConf().setAppName("Movie Search Engine in Scala"))
    // Load movie plot summaries text file
    val plotSummary = sc.textFile(args(0)).map(_.toLowerCase)
    // Load movie metadata
    val movieMetadata = sc.textFile(args(1))
    // Load file containing single search terms
    val searchSingleTerms = sc.textFile(args(2)).map(_.toLowerCase).collect()
    // Load file containing multiple search terms
    val searchMultipleTerms = sc.textFile(args(3)).map(_.toLowerCase).collect()

    logger.append("Plot Summary 1 row \n------------------")
    plotSummary.take(3).foreach(x => logger.append("\n").append(x))
    logger.append("\n\nMovie metadata \n------------------")
    movieMetadata.take(2).foreach(x => logger.append("\n").append(x))

    val stopWords = List("i", "me", "my", "myself", "we", "our", "ours", "ourselves",
      "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
      "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
      "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
      "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
      "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
      "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
      "about", "against", "between", "into", "through", "during", "before", "after",
      "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
      "under", "again", "further", "then", "once", "here", "there", "when", "where",
      "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
      "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
      "very", "s", "t", "can", "will", "just", "don", "should", "now","...")

    // Split tokens inclusive of stop words
    val plotSummaryWithSW = plotSummary.map(x => (x.split("\t")(0).toInt, x.split("\t")(1).trim().split("""\W+""")))
    logger.append("\n\nPlot summary tokens inclusive of stop words \n--------------------------------")
    plotSummaryWithSW.take(1).foreach(x => logger.append("\n").append(x))

    // Split tokens exclusive of stop words
    val plotSummaryWithoutSW = plotSummary.map(x => (x.split("\t")(0).toInt, x.split("\t")(1).trim().split("""\W+""").filterNot(x => stopWords.exists(e => x.matches(e)))))
    logger.append("\n\nPlot summary tokens exclusive of stop words \n--------------------------------")
    plotSummaryWithoutSW.take(1).foreach(x => logger.append("\n").append(x))

    // Calculate total number of documents i.e. total movies
    val summaryDocCount = plotSummaryWithoutSW.count()
    logger.append("\n\nTotal documents (movies) \n" +
      "--------------------------------\n".concat(summaryDocCount.toString()))
    // Creating token pair for each movieId and term
    val tokenPair = plotSummaryWithoutSW.flatMap(row => row._2.map(token => ((row._1, token), 1)))
    logger.append("\n\nSample tokenPair generated \n--------------------------------")
    tokenPair.take(5).foreach(x => logger.append("\n").append(x))
    //logger.saveAsTextFile(args(2))

    // Calculate term freqency in each document
    val tf = tokenPair.reduceByKey((x, y) => x + y)
    logger.append("\n\nSample tf generated \n--------------------------------")
    tf.take(5).foreach(x => logger.append("\n").append(x))

    // Calculate document frequency
    val df = tokenPair.reduceByKey((x,y) => x).map(x => (x._1._2, x._2)).reduceByKey((x,y) => x + y)
    logger.append("\n\nSample df generated \n--------------------------------")
    df.take(5).foreach(x => logger.append("\n").append(x))

    // idf = logger(|docCount|+1/docFreq)
    // Calculating idf for each term
    val idf = tokenPair.reduceByKey((x,y) => x).map(x => (x._1._2, x._2)).join(df).map(y => (y._1,y._2._2)).map(x => (x._1, math.log(summaryDocCount + 1/x._2)))
    logger.append("\n\nIDF calculated based on the formula : " +
      "idf = logger(|docCount|+1/docFreq) \n Sample idf records \n--------------------------------")
    idf.take(5).foreach(x => logger.append("\n").append(x))

    // Calculate tf-idf for each term
    val tfIdf = tf.map(x => (x._1._2, x)).join(idf).map(x => (x._2._1._1._1, x._1, x._2._1._2 * x._2._2))
    logger.append("\n\nSample tfIdf records \n--------------------------------")
    tfIdf.take(5).foreach(x => logger.append("\n").append(x))

    // Split movie metadata to have only movie Id and movie name
    val movieTitleRdd = movieMetadata.map(x => (x.split("\t")(0).toInt, x.split("\t")(2)))
    logger.append("\n\nSample records for movie meta data \n--------------------------------")
    movieTitleRdd.take(5).foreach(x => logger.append("\n").append(x))


    val searchSingleTermsCount = searchSingleTerms.count(x => true)
    searchSingleTerms.indices.foreach(idx => {
      var output1 :StringBuilder = new StringBuilder()
      output1.append("( " + idx + " ) Keyword entered : " + searchSingleTerms(idx) + "\n  Search results : \n")
      val result = tfIdf.filter(_._2.contains(searchSingleTerms(idx))).sortBy(-_._3).map(x => (x._1,1)).join(movieTitleRdd).map(y => y._2._2).distinct()
      result.take(10).foreach(x => output1.append("\n").append(x))
      output1.append("\n---------------------------------------------")
      if (searchSingleTermsCount-1 == idx) {
        sc.parallelize(output1.toString()).coalesce(1).saveAsTextFile(args(4))
      }
    })
    /*
     * In case of multiple terms in a query, we'll find the optimum results based on cosine similarity for each document
     * Cosine Similarity(query, doc) = Dot product(query, doc) / ||query|| * ||doc||
     * The document with the highest similarity will be chosen as the most relevant
     * Dot product(query, doc) = product of tf-idf of each term in query with the same term in document
     * ||query|| = Square root of sum of squares of tf-idf values of terms in the query
     * ||doc|| = Square root of sum of squares of tf-idf values of terms in the document
     */

    val searchMultipleTermsCount = searchSingleTerms.count(x => true)
    searchMultipleTerms.indices.foreach(idx => {
      var output2 :StringBuilder = new StringBuilder()
      output2.append("( " + idx + " ) Keywords entered : " + searchMultipleTerms(idx))
      val queryWords = sc.parallelize(searchMultipleTerms(idx).split(" "))
      //val queryWordsFiltered = queryWords.filterNot(t => stopWords.exists(e => t.matches(e)))
      //output2.append("\nAfter removal of stop words from query : \n")
      //queryWords.take(10).mkString(" ").foreach(x => output2.append(" ").append(x))
      val queryWordsTf = queryWords.map(x => (x,1)).reduceByKey((x, y) => x + y)
      val queryWordIdf = queryWordsTf.map(x => (x._1, math.log(2/x._2)))
      val queryWordsTfIdf = queryWordsTf.join(queryWordIdf).map(x => (x._1, x._2._1 * x._2._2))
      val queryNorm = math.sqrt(queryWordsTfIdf.fold((null, 0))((x, y) => (x._1, math.pow(x._2, 2) + math.pow(y._2, 2)))._2)
      val result = tfIdf.map(x => (x._2, (x._1,x._3))).join(queryWordsTfIdf).map(x => (x._2._1._1, (x._2._1._2 * x._2._2, x._2._1._2 * x._2._1._2))).foldByKey((0, 0))((x, y) => (x._1 + y._1, x._2 + y._2)).map(x => (x._1, x._2._1/(math.sqrt(x._2._2) * queryNorm))).sortBy(-_._2).map(x => (x._1,1)).join(movieTitleRdd).map(y => y._2._2).distinct()
      output2.append("\nSearch results : ")
      result.take(10).foreach(x => output2.append("\n").append(x))
      output2.append("\n---------------------------------------------")
      if (searchMultipleTermsCount-1 == idx) {
        sc.parallelize(output2.toString()).coalesce(1).saveAsTextFile(args(5))
      }
    })

    sc.parallelize(logger.toString()).coalesce(1).saveAsTextFile(args(6))
  }
}
