function retrive() {
    //var branch = document.getElementById("branch-select").value + "crimebranch@gmail.com";
    var branch = "chavanshrsa18it@student.mes.ac.in"
    document.getElementById("email").value = branch;

}

function loading(visibility, hidden_value) {
    console.log("Loading has been called!!")
    document.getElementById('loader').style.visibility = visibility;
    document.getElementById('search').hidden = hidden_value;
    document.getElementById('result').style.display = 'none';

}

function imgLoading(visibility){
    console.log("Loading has been called!!")

    document.getElementById('img-loader').style.visibility = visibility;

}