
(function() {
  "use strict";

  /**
   * Easy selector helper function
   */
  const select = (el, all = false) => {
    el = el.trim()
    if (all) {
      return [...document.querySelectorAll(el)]
    } else {
      return document.querySelector(el)
    }
  }

  /**
   * Easy event listener function
   */
  const on = (type, el, listener, all = false) => {
    let selectEl = select(el, all)
    if (selectEl) {
      if (all) {
        selectEl.forEach(e => e.addEventListener(type, listener))
      } else {
        selectEl.addEventListener(type, listener)
      }
    }
  }

  /**
   * Easy on scroll event listener 
   */
  const onscroll = (el, listener) => {
    el.addEventListener('scroll', listener)
  }

  /**
   * Navbar links active state on scroll
   */
  let navbarlinks = select('#navbar .scrollto', true)
  const navbarlinksActive = () => {
    let position = window.scrollY + 200
    navbarlinks.forEach(navbarlink => {
      if (!navbarlink.hash) return
      let section = select(navbarlink.hash)
      if (!section) return
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        navbarlink.classList.add('active')
      } else {
        navbarlink.classList.remove('active')
      }
    })
  }
  window.addEventListener('load', navbarlinksActive)
  onscroll(document, navbarlinksActive)

  /**
   * Scrolls to an element with header offset
   */
  const scrollto = (el) => {
    let header = select('#header')
    let offset = header.offsetHeight

    let elementPos = select(el).offsetTop
    window.scrollTo({
      top: elementPos - offset,
      behavior: 'smooth'
    })
  }

  /**
   * Toggle .header-scrolled class to #header when page is scrolled
   */
  let selectHeader = select('#header')
  let selectTopbar = select('#topbar')
  if (selectHeader) {
    const headerScrolled = () => {
      if (window.scrollY > 100) {
        selectHeader.classList.add('header-scrolled')
        if (selectTopbar) {
          selectTopbar.classList.add('topbar-scrolled')
        }
      } else {
        selectHeader.classList.remove('header-scrolled')
        if (selectTopbar) {
          selectTopbar.classList.remove('topbar-scrolled')
        }
      }
    }
    window.addEventListener('load', headerScrolled)
    onscroll(document, headerScrolled)
  }

  /**
   * Back to top button
   */
  let backtotop = select('.back-to-top')
  if (backtotop) {
    const toggleBacktotop = () => {
      if (window.scrollY > 100) {
        backtotop.classList.add('active')
      } else {
        backtotop.classList.remove('active')
      }
    }
    window.addEventListener('load', toggleBacktotop)
    onscroll(document, toggleBacktotop)
  }

  /**
   * Mobile nav toggle
   */
  on('click', '.mobile-nav-toggle', function(e) {
    select('#navbar').classList.toggle('navbar-mobile')
    this.classList.toggle('bi-list')
    this.classList.toggle('bi-x')
  })

  /**
   * Mobile nav dropdowns activate
   */
  on('click', '.navbar .dropdown > a', function(e) {
    if (select('#navbar').classList.contains('navbar-mobile')) {
      e.preventDefault()
      this.nextElementSibling.classList.toggle('dropdown-active')
    }
  }, true)

  /**
   * Scrool with ofset on links with a class name .scrollto
   */
  on('click', '.scrollto', function(e) {
    if (select(this.hash)) {
      e.preventDefault()

      let navbar = select('#navbar')
      if (navbar.classList.contains('navbar-mobile')) {
        navbar.classList.remove('navbar-mobile')
        let navbarToggle = select('.mobile-nav-toggle')
        navbarToggle.classList.toggle('bi-list')
        navbarToggle.classList.toggle('bi-x')
      }
      scrollto(this.hash)
    }
  }, true)

  /**
   * Scroll with ofset on page load with hash links in the url
   */
  window.addEventListener('load', () => {
    if (window.location.hash) {
      if (select(window.location.hash)) {
        scrollto(window.location.hash)
      }
    }
  });

  /**
   * Preloader
   */
  let preloader = select('#preloader');
  if (preloader) {
    window.addEventListener('load', () => {
      preloader.remove()
    });
  }

  /**
   * Initiate glightbox 
   */
  const glightbox = GLightbox({
    selector: '.glightbox'
  });

  /**
   * Initiate Gallery Lightbox 
   */
  const galelryLightbox = GLightbox({
    selector: '.galelry-lightbox'
  });

  /**
   * about-us slider
   */
  new Swiper('.about-us-slider', {
    speed: 600,
    loop: false,
    autoplay: {
      delay: 5000,
      disableOnInteraction: false
    },
    slidesPerView: 'auto',
    pagination: {
      el: '.swiper-pagination',
      type: 'bullets',
      clickable: true
    },
    breakpoints: {
      320: {
        slidesPerView: 1,
        spaceBetween: 20
      },

      1200: {
        slidesPerView: 2,
        spaceBetween: 20
      }
    }
  });

  /**
   * Initiate Pure Counter 
   */
  new PureCounter();

})()

/*CT TOOl */
function uploadImage() {
  const input = document.getElementById('image');
  const file = input.files[0];

  if (file) {
      const reader = new FileReader();
      reader.onload = function(e) {
          const image = new Image();
          image.src = e.target.result;
          image.onload = function() {
              const uploadedImage = document.getElementById('uploadedImage');
              uploadedImage.innerHTML = '';
              uploadedImage.appendChild(image);
          }
      }
      reader.readAsDataURL(file);
  }
}

/*login-register-forot password java*/
function switchForm(formId) {
  var loginForm = document.querySelector('.login-form');
  var registerForm = document.querySelector('.register-form');
  var createNewPasswordForm = document.querySelector('.forgot-password-form');

  if (formId === 'login-form') {
      loginForm.style.display = 'block';
      registerForm.style.display = 'none';
      createNewPasswordForm.style.display = 'none';
  } else if (formId === 'register-form') {
      loginForm.style.display = 'none';
      registerForm.style.display = 'block';
      createNewPasswordForm.style.display = 'none';
  } else if (formId === 'forgot-password-form') {
      loginForm.style.display = 'none';
      registerForm.style.display = 'none';
      createNewPasswordForm.style.display = 'block';
  }
}

/*admin*/


// Function to populate user list table
async function populateUserList(){

  const response = await fetch('/admin/get_users'); // Await the fetch response
  const jsonData = await response.json(); // Await the parsed JSON

  const users = jsonData;
  
  // const users = getUserList();
  document.getElementById('user-table').classList.remove('hidden');
  document.getElementById('view-title').classList.remove('hidden');
  document.getElementById('search').classList.remove('hidden');
  document.getElementById('add-user-form').classList.add('hidden');
  document.getElementById('title').classList.add('hidden');

  var tableBody = document.getElementById("user-table").getElementsByTagName("tbody")[0];
  tableBody.innerHTML = ""; // Clear existing rows
  if (users.length === 0) {
    // No users to display (empty search results or initial load)
    var messageRow = document.createElement("tr");
    var messageCell = document.createElement("td");
    messageCell.colSpan = 3; // Span across all columns
    messageCell.textContent = "No users found.";
    messageRow.classList.add("not-found"); // Add class for identification
    tableBody.appendChild(messageRow);
  } else {
    // Populate table with users
    for (var i = 0; i < users.length; i++) {
      var user = users[i];
      var row = document.createElement("tr");
      var userIdCell = document.createElement("td");
      userIdCell.textContent = user.user_id;
      row.appendChild(userIdCell);
      var usernameCell = document.createElement("td");
      usernameCell.textContent = user.user_name;
      row.appendChild(usernameCell);
      var emailCell = document.createElement("td");
      emailCell.textContent = user.user_email;
      row.appendChild(emailCell);

    // Create action cell with trash icon
  var actionCell = document.createElement("td");
  var deleteIcon = document.createElement("span");
  deleteIcon.classList.add("fas", "fa-trash", "clickable_icon"); // Font Awesome classes for trash icon
  
  // **Pass user ID as data attribute for easy retrieval:**
  deleteIcon.dataset.userId = user.user_id;

  deleteIcon.addEventListener("click", deleteUser); // Add click event listener
  actionCell.appendChild(deleteIcon);
  row.appendChild(actionCell);

    tableBody.appendChild(row);
  }}}

    /// ----------------- END OF POPULATE USERS ---------------

  function searchUsers() {
    const search = document.getElementById("search"); // Replace with your search input ID

    const tableBody = document.getElementById("user-table"); // Replace with your table body ID
    table_rows = tableBody.querySelectorAll("tbody tr");

    table_rows.forEach((row, i) => {
      let table_data = row.textContent,
      search_data = search.value;

      row.classList.toggle('hide',table_data.indexOf(search_data) < 0);
      })

  }

  /// ----------------- END OF SEARCH FUNCTION ---------------

async function deleteUser() {
  // Get username from button data attribute
  const userId = event.target.dataset.userId;

  // Confirmation prompt
  const message = `Are you sure you want to delete ID: ${userId} ?`
  const confirmed = await showConfirmationMessage(message);
  if (confirmed){
    try {
      // Send DELETE request to the backend API
      const response = await fetch(`/admin/delete_user/${userId}`, {
        method: 'DELETE' // Specify DELETE method for deletion
      });

      if (!response.ok) {
        throw new Error(`Failed to delete user: ${response.status}`);
      }
      general_error("User deleted successfully!");
      populateUserList();
    } catch (error){
      console.error("Error deleting user:", error);
      general_error("Error: Deletion failed. Please check the console for details."); // Informative error message
    }
  }
}

  /// ----------------- END OF DELETE USER ---------------

  function showUserForm(){
    // Remove the hidden elements
  document.getElementById('add-user-form').classList.remove('hidden');
  document.getElementById('user-table').classList.add('hidden');
  document.getElementById('view-title').classList.add('hidden');
  document.getElementById('search').classList.add('hidden');
  document.getElementById('title').classList.add('hidden');
  }
    

// Add User form submission event listener
function adminAddUser(event){
  var addUserForm = document.getElementById("add-user-form");
  event.preventDefault();

  const username = document.getElementById('admin-username').value;
  const email = document.getElementById('admin-email').value;
  const password = document.getElementById('admin-password').value;
  const confirm_password = document.getElementById('admin-confirm-password').value;

  if(password.length < 8) {  
    displayErrorMessage("Password should be at least 8 characters long!")
    return;
  }
  else if (password != confirm_password) {
    displayErrorMessage("Passwords don't match.")
    return;
  }

  const registerData = {
    "user-email": email,
    "user-name": username,
    "user-password": password,
    "confirm-password": confirm_password
  };

  // Make a POST request to the backend API for register authentication
  fetch('/admin/add_user', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(registerData)
  })
  .then(response => response.json())
  .then(data => {

    if (!data.status){
      general_error(data.message) // return the message and redirection route
    }
    else{
      displaySuccessMessage(data.message) // return the message and redirection route
    }
  })
  .catch(error => {
    // Handle authentication failure
    general_error(error.message); // return the message and redirection route
  });

  document.getElementById("admin-username").value = "";
  document.getElementById("admin-email").value = "";
  document.getElementById("admin-password").value = "";
  document.getElementById("admin-confirm-password").value = "";

}

//-----------------------------Login API---------------------------------------------


// Function to handle login form submission
function handleLogin(event) {
  event.preventDefault(); // Prevents using the default 'GET' action of form

  // Getting info from the form
  const email = document.getElementById('login-Email').value;
  const password = document.getElementById('login-password').value;
  const account_type = document.querySelector('input[name="user-type"]:checked').value;

  const loginData = {
    "login-Email": email,
    "login-password": password,
    "account_type": account_type
  };

  // Make a POST request to the backend API for login authentication
  fetch('/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(loginData)
  })
  .then(response => response.json())
  .then(data => {
    if (!data.status) {
      displayErrorMessage(data.message,'/login') // return the message and redirection route
    }
    else{
      // Redirect the user to the previous page they were trying to access
      const url = new URL(window.location.href);

      // Extract the 'next' parameter using URLSearchParams
      const params = new URLSearchParams(url.search);
      const nextParam = params.get('next');

      // Check if the parameter exists
      if (nextParam) {

        displaySuccessMessage(data.message, nextParam);
        // window.location.href = nextParam;
        // You can now use the 'nextParam' variable
      } else {
        displaySuccessMessage(data.message, '/home');
        // window.location.replace('/home')
      }
    }
  })
  .catch(error => {
    // Handle authentication failure
    displayErrorMessage(error.message,'/login'); // return the message and redirection route
  });
}

//-----------------------------Register API---------------------------------------------

function validateEmail(email) {
  const re = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z0-9\-]{1,63}\.)+[a-zA-Z]{2,63}))$/;
  return re.test(String(email).toLowerCase());
}

// Function to handle register form submission
async function handleRegister(event) {
  event.preventDefault(); // Prevents using the default 'GET' action of form

  // Getting info from the form
  const username = document.getElementById('user-name').value;
  const email = document.getElementById('user-email').value;
  const password = document.getElementById('user-password').value;
  const confirm_password = document.getElementById('confirm-password').value;
  
  // Validate email address
  const isValidEmail = validateEmail(email)
  if (!isValidEmail){
    displayErrorMessage('Please enter a valid email')
    return;
  }
  //minimum password length validation  
  else if(password.length < 8) {  
    displayErrorMessage("Password should be at least 8 characters long!")
    return;
  }
  // passwords match validation
  else if(password != confirm_password){
    displayErrorMessage("Passwords don't match!")
    return;
  }


  // Store the email in order to resend if needed
  sessionStorage.setItem('registeredEmail', email);

  const registerData = {
    "user-email": email,
    "user-name": username,
    "user-password": password,
    "confirm-password": confirm_password
  };

  // Make a POST request to the backend API for register authentication
  fetch('/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(registerData)
  })
  .then(response => response.json())
  .then(data => {
    if (!data.status){
      displayErrorMessage(data.message,'/register') // return the message and redirection route
    }
    else{
      displaySuccessMessage(data.message) // return the message and redirection route

      // Reset the form
      document.getElementById('user-name').value = "";
      document.getElementById('user-password').value = "";
      document.getElementById('user-email').value = "";
      document.getElementById('confirm-password').value = "";

      // Show resend verification link
      document.getElementById('resendVerif').classList.remove('hidden');
    }
  })
  .catch(error => {
    // Handle authentication failure
    displayErrorMessage(error.message,'/register'); // return the message and redirection route
  });
}



// Function to handle resending verification email
function handleResendVerif(event) {
  event.preventDefault(); // Prevents using the default 'GET' action of form

  const storedEmail = sessionStorage.getItem('registeredEmail');

  const resendEmail = {
    "user-email": storedEmail
  };

  // displaySuccessMessage("A new confirmation email has been sent.")

  // Make a POST request to the backend API for resending email
  fetch('/resend', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(resendEmail)
  })
  .then(response => response.json())
  .then(data => {

    if (!data.status){
    displayErrorMessage(data.message) // return the message and redirection route
  }
  else{
    displaySuccessMessage(data.message)
  }})
  .catch(error => {
    // Handle api calling failure
    displayErrorMessage(error.message); // return the message
  });

}




//------------------------- CT scan API --------------------------------

function handleCTScan(event){
  event.preventDefault(); // Prevent default form submission

  const formData = new FormData(document.getElementById('ct-scan-form'));
  console.log(formData)


  fetch('/ct_scan', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('ct-result').classList.remove('hidden');
    document.getElementById('ct-refresh').classList.remove('hidden');
    if (data.status === 1) {

      // Change the background to green if the prediction is benign or normal
      const result_div = document.getElementById('ct-result')
      if (data.prediction == 'benign' || data.prediction == 'normal'){
        result_div.style.backgroundColor ='#418C78';
      }

      // Get the analyze button element
      const element = document.getElementById('ct-button');

      // Add the class "hidden" to hide the analyze button
      element.setAttribute('class', element.getAttribute('class') + ' hidden');

      document.getElementById('ct-result').textContent = `Your prediction is: ${data.prediction}`;
    } else {
      // Get the element
      const element = document.getElementById('ct-button');

      // Add the class "hidden"
      element.setAttribute('class', element.getAttribute('class') + ' hidden');

      document.getElementById('ct-result').textContent = data.message;
    }
  })
  .catch(error => {
    console.error('Error:', error);
    // Get the element
    const element = document.getElementById('ct-button');

    // Add the class "hidden"
    element.setAttribute('class', element.getAttribute('class') + ' hidden');

    document.getElementById('ct-refresh').classList.remove('hidden');
    document.getElementById('ct-result').classList.remove('hidden');
    document.getElementById('ct-result').textContent = 'An error occurred.';
  });
}


// Remove the image on reload
window.onload = function() {
  document.getElementById('ct-image').value = "";
};




//------------------------- Survey API --------------------------------

function handleSurvey(event){
  event.preventDefault(); // Prevent default form submission

  const formData = new FormData(document.getElementById("survey-form"));


  const jsonData = {};

  // Loop through form entries and map to the desired keys
  for (const [key, value] of formData.entries()) {
    jsonData[key] = value;
  }


  fetch('/survey', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(jsonData)
  })
  .then(response => response.json())
  .then(data => {

    // Unhide/Show the survey result and refresh button
    document.getElementById('survey-result').classList.remove('hidden');
    document.getElementById('survey-refresh').classList.remove('hidden');
    if (data.status === 1) {

      // Change the background to green if the prediction is low risk
      const result_div = document.getElementById('survey-result')
      if (data.prediction == 'Low Risk'){
        result_div.style.backgroundColor ='#418C78';
      }

      // Get the submit button element
      const element = document.getElementById('survey-submit-button');

      // Add the class "hidden" to hide the submit button
      element.setAttribute('class', element.getAttribute('class') + ' hidden');

      // Add the text of the prediction to the survey result div
      document.getElementById('survey-result').textContent = `Your risk of breast cancer is: ${data.prediction}`;

      if(data.prediction == 'High Risk'){
        
        // Unhide the recommendation div container and add the styling class
        recommendationElement = document.getElementById('recommendation-container')
        recommendationElement.classList.remove('hidden');
        recommendationElement.classList.add('link-row');

        // Create a new div for text inside the recommendation container and display recommendation text
        const newDiv = document.createElement('div');
        newDiv.textContent = `Your risk assessment indicates a potentially higher risk. We commonly see this with certain factors, and based on your answers we recommend the following articles.\r\n`;
        recommendationElement.appendChild(newDiv);


        // Get the answers from the submitted survey response
        const density = jsonData['BIRADS_breast_density']
        const bmi = jsonData['bmi_group']
        const firstDegree = jsonData['first_degree_hx']

        // Create a link to the density article if the answer indicates dense breasts
        if (density == 'Heterogeneously dense' || density == 'Extremely dense'){
          const newLink = document.createElement('a');
          newLink.href = '/breast_density5';
          newLink.textContent = 'Breast Density As A Risk Factor';
          recommendationElement.appendChild(newLink);
        }

        // Create a link to the bmi article if the answer indicates high bmi
        if (bmi == '30-34.99' || bmi == '35 or more'){
          const newLink = document.createElement('a');
          newLink.href = '/bmi7';
          newLink.textContent = 'High BMI As A Risk Factor';
          recommendationElement.appendChild(newLink);
        }

        // Create a link to the family article if the answer indicates family members with breast cancer
        if (firstDegree == 'Yes'){
          const newLink = document.createElement('a');
          newLink.href = '/first_degree2';
          newLink.textContent = 'First Degree Relatives A Risk Factor';
          recommendationElement.appendChild(newLink);
        }
        
        // Create a link to the articles in general since the individual is of high risk
        const newLink = document.createElement('a');
        newLink.href = '/articles';
        newLink.textContent = 'Breast Cancer Risk Factors';
        recommendationElement.appendChild(newLink);

      }
    } else {
      // Get the submit button element
      const element = document.getElementById('survey-submit-button');

      // Add the class "hidden" to hide the submit button in order to display text
      element.setAttribute('class', element.getAttribute('class') + ' hidden');

      // Display the response message if an error occured
      document.getElementById('survey-result').textContent = data.message;
    }
  })
  .catch(error => {
    console.error('Error:', error);
    // Get the element
    const element = document.getElementById('survey-submit-button');

    // Add the class "hidden"
    element.setAttribute('class', element.getAttribute('class') + ' hidden');

    document.getElementById('survey-refresh').classList.remove('hidden');
    document.getElementById('survey-result').classList.remove('hidden');
    document.getElementById('survey-result').textContent = 'An error occurred.';
  });
}



//------------------------- Change Password API --------------------------------

// Function to handle change password form request
function handleChangePassword(event) {
  event.preventDefault(); // Prevents using the default 'GET' action of form

  // Getting info from the form
  const email = document.getElementById('change-password-user-email').value;

  const changePasswordData = {
    "user-email": email
  };

  // Make a POST request to the backend API for register authentication
  fetch('/new_password', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(changePasswordData)
  })
  .then(response => response.json())
  .then(data => {

    if (!data.status){

      displayErrorMessage(data.message) // return the message and redirection route
      document.getElementById('change-password-user-email').value = "";
    }
    else{
      displaySuccessMessage(data.message) // return the message and redirection route
      document.getElementById('change-password-user-email').value = "";
    }
  })
}


// Function to check if the passwords are valid before submission
function verifyPassword(event) {  
  var pw = document.getElementById("new-password").value;  
  var confirmPw = document.getElementById("confirm-reset-password").value;
   
  //minimum password length validation  
  if(pw.length < 8) {  
    event.preventDefault();
    displayErrorMessage("Password should be at least 8 characters long!")
  }  
//Check if passwords match
  else if(pw !== confirmPw) {
     event.preventDefault();
     displayErrorMessage("Passwords don't match!")
     document.getElementById("confirm-reset-password").value = "";
  }

}  


//---------------------------messages-----------------------------------

// Function to display success message
function displaySuccessMessage(message, redirectUrl) {
  const customAlert = document.getElementById("custom-alert");
  const alertMessage = document.querySelector("#custom-alert .alert-message");
  const closeButton = document.getElementById("close-alert");

  alertMessage.textContent = message;
  alertMessage.style.color = "#3c763d"; /* Green text for success */
  customAlert.classList.remove("hidden"); // Show the alert


  closeButton.addEventListener("click", () => {
    customAlert.classList.add("hidden"); // Hide the alert on close button click
    if (redirectUrl) {
      // window.location.href = redirectUrl; // Redirect if URL provided
      window.location.assign(redirectUrl, true);
    }
    else{
      customAlert.classList.add("hidden");
    }
  });
  // alert(message);
}

// Function to display error message
function displayErrorMessage(message, redirectUrl) {
  const customAlert = document.getElementById("custom-alert");
  const alertMessage = document.querySelector("#custom-alert .alert-message");
  const closeButton = document.getElementById("close-alert");

  alertMessage.textContent = message;
  customAlert.classList.remove("hidden"); // Show the alert

  
  closeButton.addEventListener("click", () => {
    customAlert.classList.add("hidden"); // Hide the alert on close button click
    if (redirectUrl) {
      // window.location.href = redirectUrl; // Redirect if URL provided
      window.location.assign(redirectUrl,  true)
    }
    else{
      customAlert.classList.add("hidden");
    }
  });
  // alert(message);
}

function general_error(message){
  const customAlert = document.getElementById("custom-alert");
  const alertMessage = document.querySelector("#custom-alert .alert-message");
  const closeButton = document.getElementById("close-alert");

  alertMessage.textContent = message;
  customAlert.classList.remove("hidden"); // Show the alert
  closeButton.addEventListener("click", () => {
    customAlert.classList.add("hidden"); // Hide the alert on close button click
  });

}


function showConfirmationMessage(message) {
  return new Promise((resolve, reject) => {
    // Retrieve the confirmation message element
    const confirmMessage = document.getElementById("confirm-message");

    // Set the message content
    confirmMessage.textContent = message;

    function handleButtonClick(isConfirmed) {
      // Hide the message after click
      confirmMessage.style.display = "none";

      // Resolve the promise with the confirmation value
      resolve(isConfirmed);
    }

    // Create buttons and add event listeners
    const confirmButton = document.createElement("button");
    confirmButton.textContent = "Yes";
    confirmButton.addEventListener("click", function() {
      handleButtonClick(true);
    });

    const cancelButton = document.createElement("button");
    cancelButton.textContent = "Cancel";
    cancelButton.addEventListener("click", function() {
      handleButtonClick(false);
    });

    // Add buttons to confirmation message
    const buttonContainer = document.createElement("div");
    buttonContainer.classList.add("button-container");
    buttonContainer.appendChild(confirmButton);
    buttonContainer.appendChild(cancelButton);
    confirmMessage.appendChild(buttonContainer);

    // Make the message visible
    confirmMessage.style.display = "block";
  });
}