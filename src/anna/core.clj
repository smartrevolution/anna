(ns anna.core
  (:use [clojure.core.matrix]
        [clojure.pprint])
  (:require [clojure.core.matrix.operators :as Mtrx])
  (:gen-class))

(def activation-fn (fn [x] (Math/tanh x)))
(def deactivation-fn (fn [y] (Mtrx/- 1.0 (Mtrx/* y y))))
(def learning-rate 0.2)
(def max-iterations 300)
(def error-threshold 0.005)

(defn zeros
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] 0)))

(defn ones
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] 1)))

(defn randos
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] (rand 1))))

(defn matrix-mult
  [a b]
  #_{:pre [(let [[m n] (shape a)
               [p q] (shape b)]
           (= n p))]}
  ;(println "a" (shape a) (matrix? a)) (pm a)
  ;(println "b" (shape b) (matrix? b)) (pm b)
  (let [result (mmul a b)]
    (if (matrix? result)
      result
      (matrix [result]))))

(defn bind-bias
  "Add a bias node to an activations vector"
  [activations]
  (into (matrix [[1]]) activations))

(defn mse
  [errors]
  (reduce + 0 (flatten (pow errors 2))))

(defprotocol Forwardpropagation
  (calc-activations [this input]))

(defprotocol Backpropagation
  (calc-output-delta [this ideal-output])
  (calc-hidden-delta [this weights delta])
  (calc-new-weights [this learning-rate activations]))

(defprotocol ANN
  (forwardpropagation [this input])
  (backpropagation [this ideal-output])
  (update-weights [this training-data])
  (train [this training-data])
  (exec [this input]))

(defrecord Layer [weights activations delta]
  Forwardpropagation
  (calc-activations
    [this input]
    (let [weighted-sum (matrix-mult weights (bind-bias input))
          new-activations (emap activation-fn weighted-sum)]
      (assoc this :activations new-activations)))
  Backpropagation
  (calc-output-delta
    [this ideal-output]
    (let [activations (:activations this)
          output-delta (Mtrx/* (Mtrx/- ideal-output activations)
                               (deactivation-fn activations))]
      (assoc this :delta output-delta)))
  (calc-hidden-delta
    [this weights delta]
    (let [hidden-delta (Mtrx/* (matrix-mult (transpose weights)
                                            delta)
                               (deactivation-fn (bind-bias (:activations this))))]
      ;call (rest hidden-delta) to remove the bias from the result
      (assoc this :delta (rest hidden-delta))))
  (calc-new-weights
    [this learning-rate activations]
    (let [weights-delta (matrix-mult (:delta this)
                                     (transpose
                                      (bind-bias activations)))
          new-weights (Mtrx/- (:weights this)
                              (Mtrx/* learning-rate
                                      weights-delta))]
      (assoc this :weights new-weights))))

(defrecord NeuralNetwork [layers]
  ANN
  (forwardpropagation
    [this input]
    (loop [layers (:layers this)
           accu []
           input input]
      (if (empty? layers)
        (assoc this :layers accu)
        (let [cur-layer (first layers)
              remaining-layers (rest layers)
              updated-layer (calc-activations cur-layer input)
              new-activations (:activations updated-layer)]
          (recur remaining-layers
                 (conj accu updated-layer)
                 new-activations)))))
  (backpropagation
    [this ideal-output]
    (let [reversed-layers (reverse (:layers this))
          output-layer (first reversed-layers)
          new-output-layer (calc-output-delta output-layer ideal-output)]
      (loop [layers' (rest reversed-layers)
             accu [new-output-layer]
             prev-layer new-output-layer]
        (if (empty? layers')
          (assoc this :layers (reverse accu))
          (let [cur-layer (first layers')
                remaining-layers (rest layers')
                updated-layer (calc-hidden-delta cur-layer
                                                 (:weights prev-layer)
                                                 (:delta prev-layer))]
            (recur remaining-layers
                   (conj accu updated-layer)
                   updated-layer))))))
  (update-weights
    [this input]
    (loop [layers (:layers this)
           prev-activations input
           accu []]
      (if (empty? layers)
        (assoc this :layers accu)
        (let [cur-layer (first layers)
              updated-layer (calc-new-weights cur-layer
                                              learning-rate
                                              prev-activations)]
          (recur (rest layers)
                 (:activations cur-layer)
                 (conj accu updated-layer))))))
  (train
    [this training-data]
    (loop [iter max-iterations
           outer-nn this]
      (if (= iter 0)
        outer-nn
        (recur
         (dec iter)
         (loop [training (first training-data)
                remaining-data (rest training-data)
                inner-nn outer-nn
                error 0]
           (println training error)
           (if (empty? remaining-data)
             inner-nn
             (let [nn1 (forwardpropagation inner-nn (:input training))
                   nn2 (backpropagation nn1 (:output training))
                   nn3 (update-weights nn2 (:input training))]
               (recur (first remaining-data)
                      (rest remaining-data)
                      nn3
                      (+ error
                         (mse (-> nn3 :layers last :delta)))))))))))
  (exec
    [this input]
    (-> (forwardpropagation this input) :layers last :activations)))

(defn make-neuralnetwork
  ([nodes-in-layer]
      {:pre [(and
              (vector? nodes-in-layer)
              (not (empty? nodes-in-layer))
              (> (count nodes-in-layer) 1))]}
      (let [synapses (partition 2 1 nodes-in-layer)
            layers (map (fn [[cols rows]]
                          (let [col-with-bias (inc cols)]
                            ;add one column for the bias weights
                            (Layer. (matrix (randos rows col-with-bias)) 
                                    (matrix (zeros rows 1))
                                    (matrix (zeros rows col-with-bias)))))
                        synapses)]
        (NeuralNetwork. layers)))
  ([nodes-in-layer & layer]
     (NeuralNetwork. layer)))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
