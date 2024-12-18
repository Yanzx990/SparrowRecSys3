package com.sparrowrecsys.online.model;

import java.util.ArrayList;

/**
 * Embedding Class, contains embedding vector and related calculation
 */
public class Embedding {
    //embedding vector
    ArrayList<Float> embVector;

    public Embedding(){
        this.embVector = new ArrayList<>();
    }

    public Embedding(ArrayList<Float> embVector){
        this.embVector = embVector;
    }

    /**
     *该方法用于向嵌入向量中添加一个新的维度（浮动数值）。调用该方法时，会将 element 添加到 embVector 中。
     */
    public void addDim(Float element){
        this.embVector.add(element);
    }

    public ArrayList<Float> getEmbVector() {
        return embVector;
    }

    public void setEmbVector(ArrayList<Float> embVector) {
        this.embVector = embVector;
    }

    //calculate cosine similarity between two embeddings
    public double calculateSimilarity(Embedding otherEmb){
        if (null == embVector || null == otherEmb || null == otherEmb.getEmbVector()
                || embVector.size() != otherEmb.getEmbVector().size()){
            return -1;
        }
        //计算点积：dotProduct，两个向量每个维度的乘积求和
        double dotProduct = 0;
        //denominator1 和 denominator2，即每个向量的平方和，再开根号。
        double denominator1 = 0;
        double denominator2 = 0;
        for (int i = 0; i < embVector.size(); i++){
            dotProduct += embVector.get(i) * otherEmb.getEmbVector().get(i);
            denominator1 += embVector.get(i) * embVector.get(i);
            denominator2 += otherEmb.getEmbVector().get(i) * otherEmb.getEmbVector().get(i);
        }
        return dotProduct / (Math.sqrt(denominator1) * Math.sqrt(denominator2));
    }
}
