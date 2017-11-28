import scala.math._
import java.io._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SQLContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}

object indian{
  def main(args : Array[String]) =  {
    
  //Initialization  
  val conf = new SparkConf().setAppName("Sparktastic").setMaster("local[1]")
  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)
  
  val rawDataWithHeader = sc.textFile("indiancities.csv")
  
  //Removing Header
  val header = rawDataWithHeader.first()
  val rawData = rawDataWithHeader.filter(row => row != header)
  
  case class City(name:String, total_pop:Double, sr:Double, lit_total:Double, grads: Double)
  
  def parseCity(str: String) : City = {
    val args = str.split(",").map(x => x.substring(0, math.min(5, x.length) ))
    City(args(0).toString, args(4).toDouble, args(13).toDouble, args(10).toDouble, args(19).toDouble)
  }
  
  val parsedData = rawData.map {parseCity}
  val inputDataForModel = parsedData.map{city => Vectors.dense(
      city.total_pop, city.grads, city.lit_total, city.sr
      )}
 
  //val numClusters = 4
  val numIterations = 10
  
  //Output File
  val file = new File("output.txt")
  val bw = new BufferedWriter(new FileWriter(file))
  
  for(numClusters <- 1 until 30)
  {  
  val clusters =  KMeans.train(inputDataForModel, numClusters, numIterations)
  val cost = clusters.computeCost(inputDataForModel)
  bw.write("For number of clusters = "+numClusters.toString()+" cost is "+cost.toString()+"\n")  
  }
  bw.close()
}
}  