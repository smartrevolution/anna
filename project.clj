(defproject ANNA "0.1.0-SNAPSHOT"
  :description "ANNA - The Artificial Neural Network Application"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [net.mikera/core.matrix "0.24.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]
                 [org.clojure/tools.logging "0.3.0"]
                 [org.clojure/tools.nrepl "0.2.5"]
                 [aleph "0.3.3"]
                 [org.clojure/tools.cli "0.3.1"]
                 [com.taoensso/timbre "3.3.1"]
                 [ring "1.3.1"]
                 [ring/ring-json "0.3.1"]
                 [intervox/clj-progress "0.1.6"]
                 [opencsv-clj "2.0.1"]
                 [iota "1.1.2"]
                 #_[net.mikera/vectorz-clj "0.26.0"]
                 #_[clatrix "0.4.0"]]
  :main ^:skip-aot anna.core
  :target-path "target/%s"
  :jvm-opts ["-Xmx1g"]
  :resource-paths ["data"]
  :profiles {:uberjar {:aot :all}})