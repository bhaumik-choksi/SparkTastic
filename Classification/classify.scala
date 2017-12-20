//Author: Bhaumik D Choksi

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

object classify{
  def main(args : Array[String]) =  {
    
  //Initialization  
  val conf = new SparkConf().setAppName("Sparktastic").setMaster("local[1]")
  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)
  
  val rawDataWithHeader = sc.textFile("academic_performance.csv")
  
  //Removing header
  val header = rawDataWithHeader.first()
  val rawData = rawDataWithHeader.filter(row => row != header)
  
  //Defining Student class. Format of each tuple in RDD.
  case class Student(topic: String, raisedHands: Double, discussion: Double, class_label: String)
  
  def parseStudent(str: String): Student = {
  val args = str.split(",").map { x => x.stripLineEnd }
  Student(args(6), args(9).toDouble, args(12).toDouble, args(16))
  }
  
  val parsedData = rawData.map {parseStudent}
  
  var topicMap : Map[String, Int] = Map()
  var topic_index : Int = 0
  parsedData.map { student => student.topic }.distinct().collect().foreach { x => topicMap += (x -> topic_index); 
  topic_index+=1;
  }
  
  var outputMap : Map[String, Int] = Map()
  var output_index : Int = 0
  parsedData.map { student => student.class_label }.distinct().collect().foreach{x=> outputMap += (x -> output_index);
  output_index+=1;
  }
  
  val inputDataForModel = parsedData.map { student => LabeledPoint(outputMap(student.class_label), Vectors.dense(topicMap(student.topic), 
      student.raisedHands, student.discussion))}
  
  var categoricalFeaturesInfo = Map[Int, Int]()
  categoricalFeaturesInfo += (0 -> topicMap.size)
  
  val numClasses = outputMap.size
  
  val impurity = "gini"
  val maxDepth = 5
  val maxBins = 7000
  
  val model = DecisionTree.trainClassifier(inputDataForModel, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)
	  
	  
  println(model.toDebugString)
  
  }
}