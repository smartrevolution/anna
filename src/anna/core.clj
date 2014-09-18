(ns anna.core
  (:use [clojure.core.matrix]
        [clojure.pprint])
  (:require [clojure.core.matrix.operators :as Mtrx])
  (:gen-class))


(defn g
  [z]
  (emap #(/ 1 (+ 1 (Math/exp (- %)))) z)) ;TODO: Vectorized version

(defn g'
  [z]
  (Mtrx/* (g z) (Mtrx/- 1 (g z))))

(defn zeros
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] 0)))

(defn ones
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] 1)))

(defn rand-epsilon
  [epsilon]
  (- (rand (* 2 epsilon)) epsilon))

(defn randos
  [n m]
  (compute-matrix [n m]
                  (fn [_ _] (rand-epsilon 1))))

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
  (/ (reduce + 0 (flatten (pow errors 2))) (count errors)))

(defprotocol Forwardpropagation
  (calc-activations [this input]))

(defprotocol Backpropagation
  (calc-output-delta [this ideal-output])
  (calc-hidden-delta [this weights delta])
  (calc-gradients [this activations])
  (calc-new-weights [this learning-rate momentum]))

(defprotocol ANN
  (forwardpropagation [this input])
  (backpropagation [this ideal-output])
  (calc-error [this ideal-output])
  (update-gradients [this input])
  (update-weights [this])
  (train [this training-data])
  (output [this])
  (errorr [this ideal-output])
  (exec [this input]))

(defrecord Options [learning-rate
                    max-iterations
                    error-threshold
                    momentum
                    mini-batch-size])

(defrecord Layer [weights weighted-sum activations delta gradients]
  Forwardpropagation
  (calc-activations
    [this input]
    (let [weighted-sum (matrix-mult weights (bind-bias input))
          new-activations (g weighted-sum)]
      (assoc this :weighted-sum weighted-sum :activations new-activations)))
  Backpropagation
  (calc-output-delta
    [this ideal-output]
    (let [output-error (Mtrx/- (Mtrx/- (:activations this) ideal-output))
          output-delta (Mtrx/* output-error (g' (:weighted-sum this)))]
      (assoc this :delta output-delta :error output-error)))
  (calc-hidden-delta
    [this weights delta]
    (let [hidden-delta (Mtrx/* (matrix-mult (transpose weights)
                                            delta)
                               (g' (bind-bias (:weighted-sum this))))] ;TODO: Bias?
      ;call (rest hidden-delta) to remove the bias from the result
      (assoc this :delta (rest hidden-delta))))
  (calc-gradients
    [this activations]
    (let [gradients (matrix-mult (:delta this)
                                 (transpose
                                  (bind-bias activations)))]
      (assoc this :gradients gradients)))
  (calc-new-weights
    [this learning-rate momentum]
    (let [new-weights (Mtrx/+ (:weights this)
                              (Mtrx/* learning-rate
                                      (:gradients this)))]
      (assoc this :weights new-weights))))

(defrecord NeuralNetwork [layers iterations error options]
  ANN
  (forwardpropagation
    [this input]
    (loop [layers (:layers this)
           accu []
           activations input]
      (if (empty? layers)
        (assoc this :layers accu)
        (let [cur-layer (first layers)
              remaining-layers (rest layers)
              updated-layer (calc-activations cur-layer activations)
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
  (calc-error
    [this ideal-output]
    #_(mse (-> this :layers last :delta))
    (mse (errorr this ideal-output)))
  (update-gradients
    [this input]
    (loop [layers (:layers this)
           prev-activations input
           accu []]
      (if (empty? layers)
        (assoc this :layers accu)
        (let [cur-layer (first layers)
              updated-gradients (calc-gradients cur-layer
                                                prev-activations)]
          (recur (rest layers)
                 (:activations cur-layer)
                 (conj accu updated-gradients))))))
  (update-weights
    [this]
    (loop [layers (:layers this)
           accu []]
      (if (empty? layers)
        (assoc this :layers accu)
        (let [cur-layer (first layers)
              updated-layer (calc-new-weights cur-layer
                                              (:learning-rate (:options this))
                                              (:momentum (:options this)))]
          (recur (rest layers)
                 (conj accu updated-layer))))))
  (train
    [this training-data]
    (let [opt (:options this)
          mini-batch-size (:mini-batch-size opt)
          max-iterations (:max-iterations opt)
          error-threshold (:error-threshold opt)
          select-mini-batch (fn [] (repeatedly mini-batch-size
                                               #(rand-nth training-data)))]
      (loop [iter 0
             mini-batch (select-mini-batch)
             outer-nn this]
        (if (or (< (:error outer-nn) error-threshold)
                (> iter max-iterations))
          outer-nn
          (recur (inc iter)
                 (select-mini-batch)
                 (loop [sample (first mini-batch)
                        remaining-samples (rest mini-batch)
                        total-error 0
                        inner-nn outer-nn]
                   (if (empty? remaining-samples)
                     inner-nn
                     (let [nn1 (forwardpropagation inner-nn (:input sample))
                           nn2 (backpropagation nn1 (:output sample))
                           nn3 (update-gradients nn2 (:input sample))
                           avg-error (/ (/ (calc-error nn3 (:output sample))
                                           mini-batch-size) 2)
                           summed-avg-error (+ total-error avg-error)
                           nn4 (assoc nn3 :error summed-avg-error :iterations iter)]
                       (when (mod iter 10)
                         (println iter "-" sample "=" (output nn4)
                                  "Error" (errorr nn4 (:output sample))
                                  "Error:" summed-avg-error))
                       (recur (first remaining-samples)
                              (rest remaining-samples)
                              summed-avg-error
                              (update-weights nn4))))))))))
  
  (output
    [this]
    (-> this :layers last :activations))
  (errorr
    [this ideal-output]
    (Mtrx/- ideal-output (output this)))
  (exec
    [this input]
    (let [result (-> (forwardpropagation this input)
                     output)
          flattened-result (flatten result)]
      (if (= 1 flattened-result)
        flattened-result ;ann with binary output
        result)))); ann with multiple outputs

(defn make-neuralnetwork
  ([nodes-in-layer & options]
      {:pre [(and
              (vector? nodes-in-layer)
              (not (empty? nodes-in-layer))
              (> (count nodes-in-layer) 1))]}
      (let [synapses (partition 2 1 nodes-in-layer)
            layers (map (fn [[cols rows]]
                          (let [col-with-bias (inc cols)]
                            ;add column 0 for the bias weights
                            (Layer. (matrix (randos rows col-with-bias)) 
                                    (matrix (zeros rows 1))
                                    (matrix (zeros rows 1))
                                    (matrix (zeros rows col-with-bias))
                                    (matrix (randos rows col-with-bias)))))
                        synapses)]
        (NeuralNetwork. layers 0 1.0 (or options (Options. 0.7 1000 0.001 0.3 4))))))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
