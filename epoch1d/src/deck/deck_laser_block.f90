! Copyright (C) 2010-2015 Keith Bennett <K.Bennett@warwick.ac.uk>
! Copyright (C) 2009-2010 Chris Brady <C.S.Brady@warwick.ac.uk>
!
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.

MODULE deck_laser_block

  USE strings_advanced
  USE laser
  USE utilities

  IMPLICIT NONE
  SAVE

  PRIVATE
  PUBLIC :: laser_deck_initialise, laser_deck_finalise
  PUBLIC :: laser_block_start, laser_block_end
  PUBLIC :: laser_block_handle_element, laser_block_check

  TYPE(laser_block), POINTER :: working_laser
  LOGICAL :: boundary_set = .FALSE.
  INTEGER :: boundary

CONTAINS

  SUBROUTINE laser_deck_initialise

  END SUBROUTINE laser_deck_initialise



  SUBROUTINE laser_deck_finalise

  END SUBROUTINE laser_deck_finalise



  SUBROUTINE laser_block_start

    IF (deck_state == c_ds_first) RETURN

    ! Every new laser uses the internal time function
    ALLOCATE(working_laser)
    working_laser%use_time_function = .FALSE.
    working_laser%use_phase_function = .TRUE.
    working_laser%use_profile_function = .TRUE.
    working_laser%use_omega_function = .FALSE.

  END SUBROUTINE laser_block_start



  SUBROUTINE laser_block_end

    IF (deck_state == c_ds_first) RETURN

    CALL attach_laser(working_laser)
    boundary_set = .FALSE.

  END SUBROUTINE laser_block_end



  FUNCTION laser_block_handle_element(element, value) RESULT(errcode)

    CHARACTER(*), INTENT(IN) :: element, value
    INTEGER :: errcode
    REAL(num) :: dummy
    INTEGER :: io, iu

    errcode = c_err_none
    IF (deck_state == c_ds_first) RETURN
    IF (element == blank .OR. value == blank) RETURN

    IF (str_cmp(element, 'boundary') .OR. str_cmp(element, 'direction')) THEN
      IF (rank == 0 .AND. str_cmp(element, 'direction')) THEN
        DO iu = 1, nio_units ! Print to stdout and to file
          io = io_units(iu)
          WRITE(io,*) '*** WARNING ***'
          WRITE(io,*) 'Element "direction" in the block "laser" is deprecated.'
          WRITE(io,*) 'Please use the element name "boundary" instead.'
        ENDDO
      ENDIF
      ! If the boundary has already been set, simply ignore further calls to it
      IF (boundary_set) RETURN
      boundary = as_boundary_print(value, element, errcode)
      boundary_set = .TRUE.
      CALL init_laser(boundary, working_laser)
      RETURN
    ENDIF

    IF (.NOT. boundary_set) THEN
      IF (rank == 0) THEN
        DO iu = 1, nio_units ! Print to stdout and to file
          io = io_units(iu)
          WRITE(io,*) '*** ERROR ***'
          WRITE(io,*) 'Cannot set laser properties before boundary is set'
        ENDDO
        CALL abort_code(c_err_required_element_not_set)
      ENDIF
      extended_error_string = 'boundary'
      errcode = c_err_required_element_not_set
      RETURN
    ENDIF

    IF (str_cmp(element, 'amp')) THEN
      working_laser%amp = as_real_print(value, element, errcode)
      RETURN
    ENDIF

    ! SI (W/m^2)
    IF (str_cmp(element, 'irradiance') .OR. str_cmp(element, 'intensity')) THEN
      working_laser%amp = SQRT(as_real_print(value, element, errcode) &
          / (c*epsilon0/2.0_num))
      RETURN
    ENDIF

    IF (str_cmp(element, 'irradiance_w_cm2') &
        .OR. str_cmp(element, 'intensity_w_cm2')) THEN
      working_laser%amp = SQRT(as_real_print(value, element, errcode) &
          / (c*epsilon0/2.0_num)) * 100_num
      RETURN
    ENDIF

    IF (str_cmp(element, 'omega') .OR. str_cmp(element, 'freq')) THEN
      IF (rank == 0 .AND. str_cmp(element, 'freq')) THEN
        DO iu = 1, nio_units ! Print to stdout and to file
          io = io_units(iu)
          WRITE(io,*) '*** WARNING ***'
          WRITE(io,*) 'Element "freq" in the block "laser" is deprecated.'
          WRITE(io,*) 'Please use the element name "omega" instead.'
        ENDDO
      ENDIF
      CALL initialise_stack(working_laser%omega_function)
      CALL tokenize(value, working_laser%omega_function, errcode)
      working_laser%omega = 0.0_num
      working_laser%omega_func_type = c_of_omega
      CALL laser_update_omega(working_laser)
      IF (working_laser%omega_function%is_time_varying) THEN
        working_laser%use_omega_function = .TRUE.
      ELSE
        CALL deallocate_stack(working_laser%omega_function)
      ENDIF
      RETURN
    ENDIF

    IF (str_cmp(element, 'frequency')) THEN
      CALL initialise_stack(working_laser%omega_function)
      CALL tokenize(value, working_laser%omega_function, errcode)
      working_laser%omega = 0.0_num
      working_laser%omega_func_type = c_of_freq
      CALL laser_update_omega(working_laser)
      IF (working_laser%omega_function%is_time_varying) THEN
        working_laser%use_omega_function = .TRUE.
      ELSE
        CALL deallocate_stack(working_laser%omega_function)
      ENDIF
      RETURN
    ENDIF

    IF (str_cmp(element, 'lambda')) THEN
      CALL initialise_stack(working_laser%omega_function)
      CALL tokenize(value, working_laser%omega_function, errcode)
      working_laser%omega = 0.0_num
      working_laser%omega_func_type = c_of_lambda
      CALL laser_update_omega(working_laser)
      IF (working_laser%omega_function%is_time_varying) THEN
        working_laser%use_omega_function = .TRUE.
      ELSE
        CALL deallocate_stack(working_laser%omega_function)
      ENDIF
      RETURN
    ENDIF

    IF (str_cmp(element, 'profile')) THEN
      CALL initialise_stack(working_laser%profile_function)
      CALL tokenize(value, working_laser%profile_function, errcode)
      working_laser%profile = 0.0_num
      CALL laser_update_profile(working_laser)
      IF (working_laser%profile_function%is_time_varying) THEN
        working_laser%use_profile_function = .TRUE.
      ELSE
        CALL deallocate_stack(working_laser%profile_function)
      ENDIF
      RETURN
    ENDIF

    IF (str_cmp(element, 'phase')) THEN
      CALL initialise_stack(working_laser%phase_function)
      CALL tokenize(value, working_laser%phase_function, errcode)
      working_laser%phase = 0.0_num
      CALL laser_update_phase(working_laser)
      IF (working_laser%phase_function%is_time_varying) THEN
        working_laser%use_phase_function = .TRUE.
      ELSE
        CALL deallocate_stack(working_laser%phase_function)
      ENDIF
      RETURN
    ENDIF

    IF (str_cmp(element, 't_start')) THEN
      working_laser%t_start = as_time_print(value, element, errcode)
      RETURN
    ENDIF

    IF (str_cmp(element, 't_end')) THEN
      working_laser%t_end = as_time_print(value, element, errcode)
      RETURN
    ENDIF

    IF (str_cmp(element, 't_profile')) THEN
      working_laser%use_time_function = .TRUE.
      CALL initialise_stack(working_laser%time_function)
      CALL tokenize(value, working_laser%time_function, errcode)
      ! evaluate it once to check that it's a valid block
      dummy = evaluate(working_laser%time_function, errcode)
      RETURN
    ENDIF

    IF (str_cmp(element, 'pol_angle')) THEN
      working_laser%pol_angle = as_real_print(value, element, errcode)
      RETURN
    ENDIF

    IF (str_cmp(element, 'pol')) THEN
      ! Convert from degrees to radians
      working_laser%pol_angle = &
          pi * as_real_print(value, element, errcode) / 180.0_num
      RETURN
    ENDIF

    IF (str_cmp(element, 'id')) THEN
      working_laser%id = as_integer_print(value, element, errcode)
      RETURN
    ENDIF

    errcode = c_err_unknown_element

  END FUNCTION laser_block_handle_element



  FUNCTION laser_block_check() RESULT(errcode)

    INTEGER :: errcode
    TYPE(laser_block), POINTER :: current
    INTEGER :: error, io, iu

    errcode = c_err_none

    error = 0
    current => laser_x_min
    DO WHILE(ASSOCIATED(current))
      IF (current%omega < 0.0_num) error = IOR(error, 1)
      IF (current%amp < 0.0_num) error = IOR(error, 2)
      current => current%next
    ENDDO

    current => laser_x_max
    DO WHILE(ASSOCIATED(current))
      IF (current%omega < 0.0_num) error = IOR(error, 1)
      IF (current%amp < 0.0_num) error = IOR(error, 2)
      current => current%next
    ENDDO

    IF (IAND(error, 1) /= 0) THEN
      IF (rank == 0) THEN
        DO iu = 1, nio_units ! Print to stdout and to file
          io = io_units(iu)
          WRITE(io,*) '*** ERROR ***'
          WRITE(io,*) 'Must define a "lambda" or "omega" for every laser.'
        ENDDO
      ENDIF
      errcode = c_err_missing_elements
    ENDIF

    IF (IAND(error, 2) /= 0) THEN
      IF (rank == 0) THEN
        DO iu = 1, nio_units ! Print to stdout and to file
          io = io_units(iu)
          WRITE(io,*) '*** ERROR ***'
          WRITE(io,*) 'Must define an "amp" or "irradiance" for every laser.'
        ENDDO
      ENDIF
      errcode = c_err_missing_elements
    ENDIF

  END FUNCTION laser_block_check



  FUNCTION as_time(value, err)

    CHARACTER(LEN=*), INTENT(IN) :: value
    INTEGER, INTENT(INOUT) :: err
    REAL(num) :: as_time

    IF (str_cmp(value, 'start')) THEN
      as_time = 0.0_num
      RETURN
    ENDIF

    IF (str_cmp(value, 'end')) THEN
      as_time = t_end
      RETURN
    ENDIF

    as_time = as_real(value, err)

  END FUNCTION as_time



  FUNCTION as_time_print(str_in, element, err) RESULT(res)

    CHARACTER(*), INTENT(IN) :: str_in, element
    INTEGER, INTENT(INOUT) :: err
    REAL(num) :: res

    res = as_time(str_in, err)

    IF (.NOT.print_deck_constants .OR. rank /= 0) RETURN

    WRITE(du,'(A,G18.11)') TRIM(element) // ' = ', res

  END FUNCTION as_time_print

END MODULE deck_laser_block
