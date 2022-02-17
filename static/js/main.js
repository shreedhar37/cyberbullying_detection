function retrive() {
    //var branch = document.getElementById("branch-select").value + "crimebranch@gmail.com";
    var branch = "meshree4@gmail.com"
    document.getElementById("email").value = branch;

}

function loading(visibility, hidden_value) {
    console.log("Loading has been called!!")
    document.getElementById('loader').style.visibility = visibility;
    document.getElementById('search').hidden = hidden_value;

}

