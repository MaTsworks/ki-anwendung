After clone the repository, you can create your own virtual environment and install the dependencies using:

#
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
#


Die Dockerfile-Idee entstand, um das Modelltraining auf Cloud-Diensten zu ermöglichen,
insbesondere für große Datensätze. Dadurch sollte eine skalierbare und 
flexible Umgebung geschaffen werden, die sich leicht auf leistungsstarken Servern ausführen lässt.

Allerdings wurde dieser Ansatz verworfen, da er 
1. nicht mehr benötigt wird 
2. localTraning auf kleine datasets effizienter sind (für Uni)
3. die Cloud-Kosten zu hoch wären. 

Daher kann die Dockerfile für das aktuelle Vorhaben ignoriert werden.